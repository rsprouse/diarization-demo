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

    dir1 : path-like
    The source directory to be mirrored.

    dir2 : path-like
    The target directory where new subdirectories will be created.
    '''
    dir1 = Path(dir1)
    dir2 = Path(dir2)
    proc = subprocess.run(
        ['find', '.', '-type', 'd'],
        cwd=dir1,
        capture_output=True,
        text=True
    )
    for name in proc.stdout.splitlines():
        (dir2 / Path(name)).mkdir(parents=True, exist_ok=True)

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