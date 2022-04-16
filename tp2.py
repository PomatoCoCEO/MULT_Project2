import numpy as np
import os
import warnings
import librosa
import scipy

SR = 22050
MONO = True


def import_csv(filename:str) -> np.ndarray:
    ans = np.genfromtxt(filename, skip_header = 1, delimiter=',')
    return ans [:,1:-1] # returns all feture values but the name and the quadrant

def normalize_features(matrix: np.ndarray) -> np.ndarray:
    maxs = np.max(matrix, axis = 0)
    mins = np.min(matrix, axis = 0)
    ans= np.array(matrix)
    ans[np.arange(len(matrix)),:] = (ans[np.arange(len(matrix)),: ] - mins)/(maxs-mins)
    return ans

def export_csv(filename:str,  array:np.ndarray) -> None:
    np.savetxt(filename, array, delimiter =',')

def calculate_stats(array):
    mean= np.mean(array)
    std =  np.std(array)
    median = np.median(array)
    skewness = scipy.stats.skew(array)
    kurtosis = scipy.stats.kurtosis(array)
    maximum = np.max(array)
    minimum = np.min(array)

    return np.array([mean,std,skewness, kurtosis,median, maximum, minimum])

def features_librosa(dirname:str) -> np.ndarray:    
    ans=np.array([])
    i=0
    for filename in os.listdir(dirname):
        print(filename)
        i+=1
        print(i)
        # spectral features
        y,fs = librosa.load(dirname+'/'+filename, sr = SR, mono = MONO)
        mfccs = librosa.feature.mfcc(y, sr=SR, n_mfcc=13)
        # print('Shape of mfccs: ', mfccs.shape)
        # mfccs_stats = np.zeros((mfccs.shape[0], 7))  # 7 metrics
        # mfccs_stats[np.arange(mfccs.shape[0])] = calculate_stats(mfccs[np.arange(mfccs.shape[0])]) 
        # mfccs_stats = np.array([calculate_stats(c) for c in mfccs])
        spcentroid = librosa.feature.spectral_centroid(y, sr=SR)
        spband = librosa.feature.spectral_bandwidth(y, sr=SR)
        spcontrast = librosa.feature.spectral_contrast(y, sr=SR)
        spflatness = librosa.feature.spectral_flatness(y)
        sprolloff = librosa.feature.spectral_rolloff(y, sr=SR)
        rms = librosa.feature.rms(y)
        zcr = librosa.feature.zero_crossing_rate(y)
        f0 = librosa.yin(y, sr=SR, fmin=20, fmax=11025)
        print( "f0 shape ",f0.shape)
        f0[f0==11025]=0
        all_features_array = np.vstack((mfccs, spcentroid, spband, spcontrast, spflatness, sprolloff, f0, rms, zcr))
        print( "all shape ",all_features_array.shape)
        all_stats = np.apply_along_axis(calculate_stats, 1, all_features_array).flatten()
        print( "all shape 2 ",all_stats.shape)


        tempo = librosa.beat.tempo(y,sr=SR)
        aid = np.append(all_stats, tempo)
        print(aid)
        print("shape aid ", aid.shape)
        if i==1:
            ans = np.array(aid)
        else:
            ans= np.vstack((ans,aid))
    print("ans: ",ans)
    print("ans shape: ",ans.shape)
    ans = np.array(ans)
    return normalize_features(ans)

def save_normalized_features():
    data = import_csv("dataset/top100_features.csv")
    n_data = normalize_features(data)
    print(n_data)
    print('shape: ', n_data.shape)
    export_csv('dataset/normalized_features.csv', n_data)
    return n_data


def main() -> None:
    warnings.filterwarnings("ignore")
    # save_normalized_features()
    features_norm_obtained = features_librosa('dataset/allSongs')

    export_csv('dataset/song_features.csv', features_norm_obtained)


if __name__ == "__main__":

    main()

