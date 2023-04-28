import librosa
import numpy as np
import sys
import numpy as np
import scipy.linalg
import scipy.stats

import warnings

#revised
major = np.asarray([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
minor = np.asarray([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])

def chromaextract(path):
  y, sr = librosa.load(path, res_type='kaiser_fast')
  y_harmonic = librosa.effects.hpss(y)[0]
  chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
  return chroma

def pitchdistextract(chroma):
  dist = np.zeros(12)
  notes = chroma.argmax(axis=0)
  for i in range(len(notes)):
    for k in range(12):
      if notes[i] == k:
        dist[k] += 1
  dist = dist.astype(int)
  return dist

def step_dist(x):
  y = x[1:]
  y = np.append(y, x[0])
  return y

def notedistinit(dist):
  nodedist = np.zeros((12,12))
  dist_0 = dist
  for i in range(11):
      nodedist[i] = dist_0
      dist_0 = step_dist(dist_0)
  return nodedist

def notecorr(major, minor, nodedist):
  corr_raw = np.zeros(24)
  for i in range(11):
    corr_mat_major = np.corrcoef(nodedist[i], major)
    corr_raw[i] = corr_mat_major[1][0]
    corr_mat_minor = np.corrcoef(nodedist[i], minor)
    corr_raw[i + 12] = corr_mat_minor[1][0]
  return corr_raw


def key_detection_krumhansl_shmuckler(path_audio_file):

    song = path_audio_file
    dist = pitchdistextract(chromaextract(song))
    corr_raw = notecorr(major, minor, notedistinit(dist))
    prediction = corr_raw.argsort()[-3:][::-1]

    note_val = ['C major', 'C# major', 'D major', 'D# major', 'E major', 'F major',
                'F# major', 'G major', 'G# major', 'A major', 'A# major', 'B major',
                'C minor', 'C# minor', 'D minor', 'D# minor', 'E minor', 'F minor',
                'F# minor', 'G minor', 'G# minor', 'A minor', 'A# minor', 'B minor']
    
    #print('top 3 candidates(best-to-worst):', note_val[prediction[0]], note_val[prediction[1]], note_val[prediction[2]])
    return note_val[prediction[0]]

def main():
    print(key_detection_krumhansl_shmuckler("music/happinessisawarmgun.mp3"))

if (__name__ == "__main__"):
   main()