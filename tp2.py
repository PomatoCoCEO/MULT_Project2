import numpy as np
import os
import warnings
import librosa
import scipy
from scipy.fft import fft
from scipy.fftpack import dct

SR = 22050
MONO = True
NUM_MAT = 6
N_SONGS = 900
DEBUG = False


def import_csv(filename:str) -> np.ndarray:
    ans = np.genfromtxt(filename, skip_header = 1, delimiter=',')
    return ans [:,1:-1] # returns all feture values but the name and the quadrant

def normalize_features(matrix: np.ndarray) -> np.ndarray:
    maxs = np.max(matrix, axis = 0)
    mins = np.min(matrix, axis = 0)
    ans= np.array(matrix)
    ans[np.arange(len(matrix)),:] = (ans[np.arange(len(matrix)),: ] - mins)/(maxs-mins)
    return ans

def export_csv(filename:str,  array:np.ndarray, fmt='%.18e') -> None:
    np.savetxt(filename, array, delimiter =',', fmt=fmt)

def calculate_stats(array:np.ndarray):
    mean= np.mean(array)
    std =  np.std(array)
    median = np.median(array)
    skewness = scipy.stats.skew(array)
    kurtosis = scipy.stats.kurtosis(array)
    maximum = np.max(array)
    minimum = np.min(array)

    return np.array([mean,std,skewness, kurtosis,median, maximum, minimum])

def mfccs(sample: np.ndarray, sr:int, n_mfccs:int=13, frame_length= 2048, hop_length=512):
    sample_size= len(sample)
    n_windows = (sample_size)// hop_length + 1
    begs= np.arange(n_windows)*hop_length - frame_length//2
    begs[begs<0]=0
    ends= np.arange(n_windows)*hop_length + frame_length//2
    ends[ends>sample_size]= sample_size

    delta_f = sr / frame_length
    arr_mfccs = []
    f_max: float = sr/2
    mel_max: float = hertz_to_mel(f_max)
    window_centers = (np.arange(1,41)/41) * mel_max
    window_centers = mel_to_hertz(window_centers)

    for i in range(len(begs)):
        sample_to_dft: np.ndarray = sample[begs[i]:ends[i]] * np.hanning(ends[i]-begs[i])
        dft_sample = fft(sample_to_dft)
        dft_magnitude = np.abs(dft_sample)[:len(dft_sample)//2+1] # dft coefficients from 0 to 1024 * delta_f
        log_fccs = []
        for j in range(40):
            inf_freq:float = 0
            sup_freq:float = f_max
            mid_freq = window_centers[j]
            if j!=0:
                inf_freq = window_centers[j-1]
            if j<39:     
                sup_freq = window_centers[j+1]
            pos_inf_freq = int(np.ceil(inf_freq / delta_f))
            pos_sup_freq = int(np.floor(sup_freq / delta_f))
            total_cc = 0
            for k in range(pos_inf_freq, min(len(dft_magnitude),pos_sup_freq+1)):
                freq_k = k*delta_f
                coef:float = 0
                if freq_k < mid_freq:
                    perc = (freq_k -inf_freq)/(mid_freq-inf_freq)
                    coef = perc
                else:
                    perc = (freq_k - mid_freq)/(sup_freq-mid_freq)
                    coef = 1-perc
                total_cc += coef * dft_magnitude[k]
            log_total_cc = np.log10(total_cc)
            log_fccs.append(log_total_cc)
        log_fccs = np.array(log_fccs)
        dft_coefs = dct(log_fccs)[:n_mfccs]
        arr_mfccs.append(dft_coefs)
    ret = np.array(arr_mfccs).T
    print("ret shape: ",ret.shape)
    # 961413169
    return ret

def features_librosa(dirname:str) -> np.ndarray:    
    ans=np.array([])
    i=0
    for filename in os.listdir(dirname):
        print_debug(filename)
        i+=1
        # spectral features
        y,fs = librosa.load(dirname+'/'+filename, sr = SR, mono = MONO)
        mfccs_file = mfccs(y, SR)# librosa.feature.mfcc(y, sr=SR, n_mfcc=13)
        spcentroid = librosa.feature.spectral_centroid(y, sr=SR)
        spband = librosa.feature.spectral_bandwidth(y, sr=SR)
        spcontrast = librosa.feature.spectral_contrast(y, sr=SR)
        spflatness = librosa.feature.spectral_flatness(y)
        sprolloff = librosa.feature.spectral_rolloff(y, sr=SR)
        rms = librosa.feature.rms(y)
        zcr = librosa.feature.zero_crossing_rate(y)
        f0 = librosa.yin(y, sr=SR, fmin=20, fmax=11025)
        f0[f0==11025]=0
        all_features_array = np.vstack((mfccs_file, spcentroid, spband, spcontrast, spflatness, sprolloff, f0, rms, zcr))
        all_stats = np.apply_along_axis(calculate_stats, 1, all_features_array).flatten()


        tempo = librosa.beat.tempo(y,sr=SR)
        aid = np.append(all_stats, tempo)
        # print_debug(aid)
        if i==1:
            ans = np.array(aid)
        else:
            ans= np.vstack((ans,aid))
    # print_debug("ans: ",ans)
    # print_debug("ans shape: ",ans.shape)
    ans = np.array(ans)
    return normalize_features(ans)

def save_normalized_features() -> np.ndarray:
    data = import_csv("dataset/top100_features.csv")
    n_data = normalize_features(data)
    print_debug(n_data)
    # print_debug('shape: ', n_data.shape)
    export_csv('dataset/normalized_features.csv', n_data)
    return n_data

def print_debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

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
    return distance_mat

def get_distance_matrices():
    song_features = np.genfromtxt('dataset/song_features.csv', skip_header = 0, delimiter=',') 
    # the skip header must be set to zero!!!
    top100 = np.genfromtxt('dataset/normalized_features.csv', skip_header = 0, delimiter=',')
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

def get_query_ranking(filename, index, distance_matrices, all_songs)-> np.ndarray:
    print_debug( "Music: ", filename)
    ans = []
    for i in range(len(distance_matrices)):
        line=distance_matrices[i,index]
        sorted_distances = np.argsort(line)
        indices = sorted_distances[1:21]
        ans.append(indices)
        if DEBUG:
            print_debug("According to metric ",i)
            for j in indices:
                print_debug(all_songs[j])
            print_debug("------")
    ans = np.array(ans)
    return ans

def score_distance(md1:np.ndarray, md2:np.ndarray)->int:
    ans=0
    inds = [1,3]
    sps = [9,11]
    for i in inds:
        if md1[i] == md2[i]:
            ans +=1
    for s in sps:
        # print_debug("s = ",s, " and md1[s] = ", md1[s], " and md2[s] =",md2[s] )
        sp1:list = md1[s].split("; ")
        sp2:list = md2[s].split("; ")
        print_debug("sp1 len is ", len(sp1), "and len sp2 is", len(sp2))
        for str_prop in sp1:
            if str_prop in sp2:
                ans +=1
    return ans    

def get_metadata_ranking(filename:str, index:int, all_songs:list, metadata_matrix:np.ndarray)->np.ndarray:
    print_debug("Music: ", filename)
    md_file = metadata_matrix[index]
    rating = np.zeros((len(all_songs)))
    for i in range(len(all_songs)):
        md_i = metadata_matrix[i]
        rating[i] = score_distance(md_i, md_file)
        print_debug("Score: ",rating[i])
    sorted_dists = np.argsort(rating)
    sorted_dists = np.flip(sorted_dists)
    reccoms = sorted_dists[1:21] # the first 20 songs that are supposed to be considered for the ranking
    return reccoms

def get_all_metadata_rankings(all_songs: list, metadata_matrix: np.ndarray, queries:list)->np.ndarray:
    a:list =[]
    for q in queries:
        index:int  = all_songs.index(q)
        a.append(get_metadata_ranking(q,index, all_songs, metadata_matrix))
    a = np.array(a)
    return a

def get_rankings(distance_matrices, all_songs, queries)->np.ndarray:
    rks=[]
    for q in queries:
        index = all_songs.index(q)
        rks.append(get_query_ranking ( q,index, distance_matrices, all_songs))
    rks = np.array(rks)
    a,b,c = rks.shape
    ans = np.zeros((b,a,c))
    for i in range(b):
        ans[i] = rks[:,i,:]
    return ans 

def read_distance_mats()->np.ndarray:

    arr = np.zeros((NUM_MAT, N_SONGS, N_SONGS ))

    filenames = ['euclidean_song_features','euclidean_top100','manhattan_song_features','manhattan_top100','cosine_song_features','cosine_top100',]

    for i in range(len(filenames)):
        gen =  np.genfromtxt('dataset/results/'+filenames[i]+'.csv', skip_header = 0, delimiter=',')
        arr[i] = gen
    return arr

def print_rankings_matrices(ranking_matrix:np.ndarray, all_songs:list, queries:list) -> None :
    for mat in ranking_matrix:
        print_rankings(mat, all_songs, queries)

def print_rankings(ranking_matrix:np.ndarray, all_songs:list, queries:list)->None:
    for i in range(len(queries)):
        q = queries[i]
        print("Recommendation for music", q, ":")
        for r in ranking_matrix[i]:
            print(all_songs[r])
        print("--------")

def save_feature_ranks(all_ranks:np.ndarray):
    for i in range(len(all_ranks)):
        export_csv(f"dataset/results/ranking_features_metric_{i}.csv", all_ranks[i], fmt="%d")

def hertz_to_mel(hertz:float)-> float:
    return 2595 * np.log10(1+hertz/700)

def mel_to_hertz(mel:float) -> float:
    return 700 * (np.power(10, mel/2595)-1)
# now for the implementation of MFCCs



def main() -> None:
    warnings.filterwarnings("ignore")
    # save_normalized_features()
    features_norm_obtained = features_librosa('dataset/allSongs')

    # export_csv('dataset/song_features.csv', features_norm_obtained)
    # distance_matrices = read_distance_mats() # reads distance matrices, already obtained
    # all_songs= os.listdir('dataset/allSongs')
    # queries = os.listdir('Queries')

    # all_feature_ranks = get_rankings(distance_matrices, all_songs, queries) # this gets all rankings
    # save_feature_ranks(all_feature_ranks) # the rankings are now saved in the disk
    # metadata_matrix = np.genfromtxt('dataset/panda_dataset_taffc_metadata.csv', delimiter=',',skip_header = 1, encoding = None, dtype=None)
    # metadata_rankings = get_all_metadata_rankings(all_songs, metadata_matrix, queries) # getting rankings based on metadata
    # metadata_rankings = metadata_rankings[np.newaxis, :,:]
    # all_rankings = np.vstack((metadata_rankings, all_feature_ranks)).astype(np.int32)
    print_rankings_matrices(all_rankings, all_songs, queries)
    # print_rankings(all_feature_ranks, all_songs, queries)
    print("Shapes:",all_feature_ranks.shape, "and ", metadata_rankings.shape)
    for i in range(len(all_rankings)):
        export_csv(f"dataset/results/metadata_ratings_{i}.csv", all_rankings[i,:,:], fmt="%d")


if __name__ == "__main__":

    main()

