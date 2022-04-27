import numpy as np
import os
import warnings
import librosa
import scipy

SR = 22050
MONO = True
NUM_MAT = 6
N_SONGS = 900


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

def calculate_stats(array:np.ndarray):
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

def save_normalized_features() -> np.ndarray:
    data = import_csv("dataset/top100_features.csv")
    n_data = normalize_features(data)
    print(n_data)
    print('shape: ', n_data.shape)
    export_csv('dataset/normalized_features.csv', n_data)
    return n_data

# --- week 3 ---
# functions
def euclidean_distance (vec1:np.ndarray, vec2:np.ndarray) -> float:
    return np.linalg.norm(vec1-vec2)

def manhattan_distance(vec1:np.ndarray, vec2:np.ndarray) -> float:
    return np.sum(np.abs(vec1-vec2))

def cosine_distance (vec1:np.ndarray, vec2:np.ndarray) -> float:
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))

def distance_matrix(feature_matrix:np.ndarray, distance_function, filename:str):
    lines= len(feature_matrix)
    distance_mat =  np.zeros((lines,lines))
    # distance_mat[np.arange(lines), np.arange(lines)] = distance_function(feature_matrix[np.arange(lines)], feature_matrix[np.arange(lines)])  
    feature_matrix[feature_matrix!=feature_matrix] = 0
    
    for i in range(len(feature_matrix)):
        for j in range(i+1):
            distance_mat[i,j] = distance_function(feature_matrix[i], feature_matrix[j])
            distance_mat[j,i] = distance_mat[i,j]
    export_csv(filename, distance_mat)
    print(distance_mat.shape)
    return distance_mat

def get_distance_matrices():
    song_features = np.genfromtxt('dataset/song_features.csv', skip_header = 1, delimiter=',')
    top100 = np.genfromtxt('dataset/normalized_features.csv', skip_header = 1, delimiter=',')

    d_functions = [euclidean_distance, manhattan_distance, cosine_distance]
    function_names=['euclidean','manhattan', 'cosine']
    matrices = [top100, song_features]
    matrix_names=['top100','song_features']
    n_functions= len(d_functions)
    n_mat = len(matrices)
    n_songs=  len(song_features)
    arr = np.zeros(( n_functions* n_mat,n_songs,n_songs ))
    # arr = [[[] for i in range(len(function_names))] for j in range(len(matrix_names))]
    
    for i in range(len(function_names)):
        for j in range(len(matrices)):
            arr[i*n_mat+j]= distance_matrix(matrices[j], d_functions[i], f"dataset/results/{function_names[i]}_{matrix_names[j]}.csv")
    # arr = np.array(arr)
    return arr

def get_query_ranking(filename, index, distance_matrices, all_songs):
    print( "Music: ", filename)
    for i in range(len(distance_matrices)):
        line=distance_matrices[i,index]
        sorted_distances = np.argsort(line)
        indices = sorted_distances[1:21]
        for j in indices:
            print(all_songs[j])
        print("------")

def get_rankings(distance_matrices):
    all_songs= os.listdir('dataset/allSongs')
    queries = os.listdir('Queries')
    for q in queries:
        index = all_songs.index(q)
        print(index)
        get_query_ranking ( q,index, distance_matrices, all_songs)

def read_distance_mats():

    arr = np.zeros((NUM_MAT, N_SONGS, N_SONGS ))

    filenames = ['euclidean_song_features','euclidean_top100','manhattan_song_features','manhattan_top100','cosine_song_feature','cosine_top100',]

    for i in range(len(filenames)):
        gen =  np.genfromtxt('dataset/results/'+filenames[i]+'.csv', skip_header = 0, delimiter=',')
        print("gen shape: ", gen.shape)
        arr[i] = gen
    return arr



def main() -> None:
    warnings.filterwarnings("ignore")
    # save_normalized_features()
    # features_norm_obtained = features_librosa('dataset/allSongs')

    # export_csv('dataset/song_features.csv', features_norm_obtained)

    distance_matrices = get_distance_matrices()
    
    distance_matrices = read_distance_mats()
    get_rankings(distance_matrices)


if __name__ == "__main__":

    main()

