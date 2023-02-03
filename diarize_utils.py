import os, sys, subprocess
import datetime
from pathlib import Path
from phonlab.utils import dir2df

def compare_dirs(dir1, ext1, dir2, ext2):
    '''
    Recursively compare two directories for matching files in one-to-one
    correspondence and return names from the first that do not have a
    match in the second. Files match if their relative path and filename without
    extension are identical.
    '''
    dir1 = Path(dir1)
    if ext1 != '':
        ext1 = rf'\{ext1}$' if ext1.startswith('.') else rf'\.{ext1}$'
    df1 = dir2df(dir1, fnpat=ext1, addcols=['barename'])
    
    dir2 = Path(dir2)
    if ext2 != '':
        ext2 = rf'\{ext2}$' if ext2.startswith('.') else rf'\.{ext2}$'
    df2 = dir2df(dir2, fnpat=ext2, addcols=['barename'])

    mrgdf = df1.merge(
        df2,
        on=['relpath', 'barename'],
        how='left',
        suffixes=['_1', '_2']
    )
    return mrgdf[mrgdf['fname_2'] \
               .isna()].rename({'fname_1': 'fname'}, axis='columns') \
               .drop('fname_2', axis='columns') \
               .reset_index(drop=True)

def mirror_dir(dir1, dir2):
    '''
    Mirror a directory's subdirectories in another directory.

    dir1: path-like
    The source directory to be mirrored.

    dir2: path-like
    The target directory where new subdirectories will be created.
    
    Notes
    -----
    This simple implementation is not efficient and may not be suitable for
    large directory structures or for use with network file systems.
    '''
    dir1 = Path(dir1)
    dir2 = Path(dir2)
    proc = subprocess.run(
        ['find', '.', '-type', 'd', '-depth'],
        cwd=dir1,
        capture_output=True,
        text=True
    )
    for name in proc.stdout.splitlines():
        (dir2 / name).mkdir(parents=True, exist_ok=True)

def prep_audio(infile, outfile, chan='-'):
    '''
    Prepare an audio file for diarization by resampling and converting to mono
    by extracting a channel or mixing down all channels.
    
    Parameters
    ----------
    
    infile: path-like
    The input audio file path. The input audio can be any file type supported by sox.
    
    outfile: path-like
    The output audio file path.
    
    chan: int or '-' (default '-')
    For a multichannel input file, the channel to extract for `outfile`. The special
    string value `-` will cause all input channels to mix down to mono. The int value
    may be specified as int or str type. For stereo the left channel is `1`, and the
    right channel is '2'.
    
    Returns
    -------
    
    No value is returned from this function. A CalledProcessError will be raised if
    the sox call fails.
    '''
    soxargs = [
        'sox', str(infile), '-r', '16000', str(outfile), 'remix', chan
    ]
    try:
        subprocess.run(soxargs, check=True)
    except subprocess.CalledProcessError as e:
        msg = f'Error while prepping audio file {infile}:\n{e}'
        raise subprocess.CalledProcessError(msg)

def diarize(wavfile, pipeline, num_spkr, rttmfile):
    '''
    Diarize a .wav file using a pyannote-audio pipeline.
    '''
    diarization = pipeline(wavfile, num_speakers=2)
    with open(rttmfile, 'w') as outfile:
        diarization.write_rttm(outfile)

def diarize_df(df, pipeline, num_spkr, wavdir, rttmdir):
    for row in df.itertuples():
        wavfile = wavdir / row.relpath / row.fname
        rttmfile = rttmdir / row.relpath / f'{row.barename}.rttm'
        try:
            diarize(wavfile, pipeline, num_spkr, rttmfile)
        except Exception as e:
            sys.stderr.write(f'Failed to diarize .wav file {wavfile}.')
            sys.stderr.write(e)

def rttm2df(rttmfile):
    '''
    Load an .rttm file into a dataframe.
    '''
    df = pd.read_csv(
        rttmfile,
        sep=' ',
        header=None,
        usecols=[1, 3, 4, 7],
        names=['fid', 't1', 'dur', 'spkr'],
        converters={'t1': lambda x: float(x) * 1000, 'dur': lambda x: float(x) * 1000},
        dtype={'fid': 'category', 'spkr': 'category'}
    )
    df['t2'] = df['t1'] + df['dur']
    return df

def buffer_tier(tier, msec):
    '''
    Buffer t1 and t2 in a tier by `msec` where possible.
    # TODO: check for proper handling of starts/ends less than buf_ms from edges
    '''
    tier['prev_t2'] = tier['t2'].shift(1, fill_value=np.max([tier.t1.min() - (2 * msec), 0]))
    tier['next_t1'] = tier['t1'].shift(-1, fill_value=tier.t2.max())
    # Check for overlapping speaker utterances. 
    assert(np.all(tier['t1'] < tier['next_t1']))
    pregap_midpt = tier.loc[:, ['t1', 'prev_t2']].mean(axis=1)
    tier['t1_buf'] = np.max(
        [tier['t1'] - msec, pregap_midpt],
        axis=0
    )
    postgap_midpt = tier.loc[:, ['t2', 'next_t1']].mean(axis=1)
    tier['t2_buf'] = np.min(
        [tier['t2'] + msec, postgap_midpt],
        axis=0
    )
    return tier

def rttm2eaf(rttms, buffer_ms=None, names={}):
    '''
    Convert left/right .rttm files to an .eaf, one tier per speaker per channel.
    Buffer t1 and t2 in a tier by `buffer_ms` where possible.
    '''
    eaf = pympi.Elan.Eaf()
    eaf.remove_tier('default')
    eaf.add_language('yid')
    for chan_s, rttmfile in rttms.items():
        df = rttm2df(rttmfile)
        for spkr in df['spkr'].cat.categories.sort_values():
            try:
                tiername = names[(chan_s, spkr)]
            except KeyError:
                tiername = f'{chan_s}_{spkr}'
            eaf.add_tier(tiername, language='yid')
            spkrdf = df[df['spkr'] == spkr].copy()
            if buffer_ms is not None:
                spkrdf = buffer_tier(spkrdf, buffer_ms)
            for row in spkrdf.itertuples():
                eaf.add_annotation(tiername, int(row.t1_buf), int(row.t2_buf))
    return eaf