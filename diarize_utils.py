import os, sys, subprocess
import datetime
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import pympi
from phonlab.utils import dir2df
from audiolabel import read_label, df2tg

def compare_dirs(dir1, ext1, dir2, ext2):
    '''
    Recursively compare two directories for matching files in one-to-one
    correspondence and return names from the first that do not have a
    match in the second. Files match if their relative path and filename
    without extension are identical.

    Parameters
    ----------

    dir1 : path-like
        The directory path that is the source of comparison.
    ext1 : str
        The filename extension used to filter results from `dir1`. Only
        filenames that end with `ext1` are included in the results.
    dir2 : path-like
        The directory path that is the target of comparison.
    ext2 : str
        The filename extension used to filter results from `dir2`.

    Returns
    -------
    dataframe
        The results dataframe contains one row per filename with
        extension `ext1` that is found in `dir1` and does *not* have a
        corresponding file with extension `ext2` found in `dir2`.
        The columns of this dataframe are `relpath`, `fname`, and `barename`.
        The `relpath` value is the relative path from `dir1` to the
        file named by `fname`. The `barename` column contains the filename
        without its extension.
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

def prep_audio(infile, outfile, chan='-'):
    '''
    Prepare an audio file for diarization by resampling and converting to mono
    by extracting a channel or mixing down all channels, using the `sox`
    commandline utility.
    
    Parameters
    ----------
    
    infile : path-like
        The input audio file path. The input audio can be any file type
        supported by `sox`.
    outfile : path-like
        The output audio file path.
    chan : int or '-' (default '-')
        For a multichannel input file, the channel to extract for `outfile`. The
        special string value `-` will cause all input channels to mix down to
        mono. The int value may be specified as int or str type. For stereo the
        left channel is `1`, and the right channel is '2'.
    
    Returns
    -------
    
    No value is returned from this function.

    Errors
    ------

    A CalledProcessError will be raised if the `sox` subprocess call fails.
    '''
    outfile.parent.mkdir(parents=True, exist_ok=True)
    soxargs = [
        'sox', str(infile), '-r', '16000', str(outfile), 'remix', str(chan)
    ]
    try:
        subprocess.run(soxargs, check=True)
    except subprocess.CalledProcessError as e:
        msg = f'Error while prepping audio file {infile}:\n{e}'
        raise subprocess.CalledProcessError(msg)

def write_eaf(dfs, tiernames, outfile, speech_label, t1col, t2col):
    '''
    Write list of dataframes as tiers of an .eaf file.

    Parameters
    ----------

    dfs : list of dataframes
        The dataframes representing annotation tiers to be written to an
        `.eaf` file.
    tiernames : list of str
        The tier names to be written to the `.eaf` file.
    outfile : path-like
        The filename to use to write the `.eaf` file.
    speech_label : str
        The string to use as the label content for the label rows in the
        dataframes in `dfs`. For diarization tasks it is normally best
        to use the empty string '' for this value for `.eaf` outputs.
    t1col : str
        The name of the column in the dataframes where the `t1` start value
        is found.
    t2col : str
        The name of the column in the dataframes where the `t2` end value
        is found.

    Returns
    -------
    
    No value is returned from this function.
    '''
    eaf = pympi.Elan.Eaf()
    eaf.remove_tier('default')
    for name, df in zip(tiernames, dfs):
        eaf.add_tier(name)
        for row in df.itertuples():
            eaf.add_annotation(
                name,
                int(getattr(row, t1col)),
                int(getattr(row, t2col)),
                value=speech_label
            )
    eaf.to_file(outfile)

def diar2df(diarization, buffer_sec, speech_label=''):
    '''
    Convert a pyannote diarization to a list of dataframes that contain
    the speech regions associated with a speaker. There is one dataframe
    per speaker.

    Parameters
    ----------

    diarization : pipeline output
        The output of a `pyannote-audio` `SpeakerDiarization` pipeline.
    buffer_sec : float
        The amount of time in seconds to pad the identified speech regions
        found by diarization. If the padding would create an overlap with
        another speech region, the midpoint of gap between the regions is used
        instead.
    speech_label : str (default '')
        The string to use as the label content for the label rows in the
        dataframes in `dfs`.

    Returns
    -------
    tuple consisting of:
        list of dataframes
            Each dataframe contains rows that represent speech regions for
            a speaker identified in the diarization.
        list of str
            The list of tier names that correspond to the dataframes.
    '''
    tiers = {}
    for segment, _, label in diarization.itertracks(yield_label=True):
        if label not in tiers.keys():
            tiers[label] = []
        tiers[label].append(
            {
                't1': segment.start,
                't2': segment.start + segment.duration,
                'text': speech_label
            }
        )
    tnames = list(tiers.keys())
    dflist = [
        buffer_tier(
            pd.DataFrame(tiers[name]),
            buffer_sec
        ) for name in tnames
    ]
    return (dflist, tnames)

def buffer_tier(tier, sec):
    '''
    Buffer t1 and t2 in a tier by `sec` where possible.

    Parameters
    ----------

    tier : dataframe
        A dataframe representing rows of speech regions.
    sec : float
        The amount of time in seconds to pad the speech regions.
        If the padding would create an overlap with another speech region,
        the midpoint of gap between the regions is used instead.

    Returns
    -------
    dataframe
        The original dataframe with new columns `t1_buf` and `t2_buf`, which
        contain the buffered times for `t1` and `t2`.
    '''
    tier['prev_t2'] = \
        tier['t2'].shift(
            1,
            fill_value=np.max([tier['t1'].min() - (2.0 * sec), 0])
        )
    tier['next_t1'] = tier['t1'].shift(-1, fill_value=tier['t2'].max())
    # Doublecheck for overlapping speaker utterances. We expect to never find these.
    assert(np.all(tier['t1'] < tier['next_t1']))
    pregap_midpt = tier.loc[:, ['t1', 'prev_t2']].mean(axis=1)
    tier['t1_buf'] = np.max(
        [tier['t1'] - sec, pregap_midpt],
        axis=0
    )
    postgap_midpt = tier.loc[:, ['t2', 'next_t1']].mean(axis=1)
    tier['t2_buf'] = np.min(
        [tier['t2'] + sec, postgap_midpt],
        axis=0
    )
    return tier

def diarize(wavfile, pipeline, outfile, num_speakers, buffer_sec, speech_label,
    fill_gaps):
    '''
    Diarize a .wav file using a pyannote-audio pipeline.

    Parameters
    ----------

    wavfile : path-like
        The input `.wav` file to diarize.
    pipeline : `pyannote-audio` `SpeakerDiarization` pipeline
        An instantiation of a `pyannote-audio` `SpeakerDiarization` pipeline.
    outfile : path-like
        The filename to use to write the diarization output. The filename must
        end with an `.eaf` (ELAN) or `.TextGrid` (Praat) extension.
    num_speakers : int
        The number of speakers to be diarized.
    buffer_sec : float
        The amount of time in seconds to pad the identified speech regions
        found by diarization. If the padding would create an overlap with
        another speech region, the midpoint of gap between the regions is used
        instead.
    speech_label : str
        The string to use as the label content for the label rows in the
        dataframes in `dfs`.
    fill_gaps : str or None
        The string value to use as the `fill_gaps` value passed to `df2tg`.
        If None, no non-speech intervals are interpolated into an output TextGrid.

    Returns
    -------
    
    No value is returned from this function.
    '''
    diarization = pipeline(wavfile, num_speakers=num_speakers)
    dfs, tiernames = diar2df(
        diarization,
        buffer_sec=buffer_sec,
        speech_label=speech_label
    )
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if outfile.suffix == '.eaf':
        # Scale seconds to milliseconds for .eaf output.
        tcols = ['t1_buf', 't2_buf']
        for df in dfs:
            df.loc[:, tcols] = df.loc[:, tcols] * 1000
        write_eaf(dfs, tiernames, outfile, speech_label, *tcols)
    elif outfile.suffix == '.TextGrid':
        dur = librosa.get_duration(filename=wavfile)
        df2tg(
            dfs,
            tnames=tiernames,
            lbl='text', t1='t1_buf', t2='t2_buf',
            start=0.0, end=dur,
            fill_gaps=fill_gaps,
            outfile=outfile
        )

def sort_tiers(annodir, chans, stereodir, relpath, fname):
    '''
    Combine left/right diarized label tiers into dataframes and sort them
    according to total duration of utterances within the tier. Also load
    stereo audio data and secondarily sort dataframes according to the
    mean sample magnitude of the audio corresponding to the utterances
    in the tier.

    Parameters
    ----------

    annodir : path-like
        Base directory where input annotation files (`.eaf` or `.TextGrid`) are
        found in per-channel subdirectories.
    chans : list of str
        The names of the channel subdirectories. This is combined with
        `annodir` to create a channel-specific path.
    stereodir : path-like
        The base directory where stereo audio input files are found.
    relpath : str
        The relative path from a channel directory to an annotation file.
    fname : str
        The filename of a `left` or `right` annotation file.

    Returns
    -------
    tuple consisting of:
        list of dataframes
            Each dataframe contains rows that represent speech regions for
            a speaker identified in the diarization.
        list of str
            The list of tier names that correspond to the dataframes.
    '''
    stereowav = (stereodir/relpath/fname).with_suffix('.wav')
    data, rate = librosa.load(stereowav, sr=None, mono=False)
    spkrchans = []
    tiers = []
    if fname.endswith('TextGrid'):
        ftype = 'praat'
        tscale = 1
    else:
        ftype = 'eaf'
        tscale = 0.001
    for cidx, chan in enumerate(chans):
        dfs = read_label(
            annodir/chan/relpath/fname,
            ftype=ftype
        )
        for df in dfs:
            # Add t1/t2 as sample indexes rather than times.
            idxdf = pd.DataFrame({
                't1idx': (df['t1'] * tscale * rate).astype(int),
                't2idx': (df['t2'] * tscale * rate).astype(int)
            })
            # Collect all sample indexes that occur during the labels.
            uttidx = np.hstack(
                [np.arange(r.t1idx, r.t2idx) for r in idxdf.itertuples()]
            )
            # Calculate duration of all labels.
            dur = (df['t2'] - df['t1']).sum()
            avgmag = np.abs(data[cidx, uttidx]).sum() / dur
            spkrchans.append(
                {'totdur': dur, 'avgmag': avgmag, 'chan': chan}
            )
            tiers.append(df)
    spkrdf = pd.DataFrame(spkrchans).sort_values('totdur')
    try:
        assert(len(spkrdf) == 4)   # Two channels, two speakers
        # Interviewer should be least active talker (shortest duration)
        interviewer = spkrdf[0:2].sort_values('avgmag')
        subject = spkrdf[2:4].sort_values('avgmag')
        # Each speaker should appear exactly once per channel.
        assert(np.all(interviewer['chan'].duplicated() == False))
        assert(np.all(subject['chan'].duplicated() == False))
        sortidx = list(interviewer.index) + list(subject.index)
        tiers = [tiers[i] for i in sortidx]
        tiernames = [
            'Interviewer unlikely', 'Interviewer probable',
            'Subject unlikely', 'Subject probable'
        ]
    except AssertionError:
        # Leave tier order unchanged and use 'unknown' names.
        tiernames = ['unknown_0', 'unknown_1', 'unknown_2', 'unknown_3']
    # Return tiers and names according to sort order.
    return tiers, tiernames
