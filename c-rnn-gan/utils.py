import math


def tone_to_freq(tone):
  """
    returns the frequency of a tone.

    formulas from
      * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
      * https://en.wikipedia.org/wiki/Cent_(music)
  """
  return math.pow(2, (tone - 69.0) / 12.0) * 440.0


def freq_to_tone(freq):
    '''
    Inverse transform of tone_to_freq
    '''
    if freq <= 0.0:
        raise ("Invalid frequncy!")
    tone = 69.0 + 12.0 * math.log(freq / 440.0, 2)
    tone = int(tone)
    return tone
