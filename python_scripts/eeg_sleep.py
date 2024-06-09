import mne
import numpy as np
import pandas as pd
from utilities import project_data_dir
from mne.io import read_raw_edf
from mne.filter import filter_data
import re
import time
from scipy.stats import kurtosis
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema, butter, filtfilt,savgol_filter,find_peaks,welch
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter
from datetime import datetime, timedelta

import yasa
from usleep_api import USleepAPI


class EEGSleep:
    def __init__(self, project_name):
        """
        Initialize the EEGSleep class.

        Parameters:
        - project_name (str): The name of the project associated with the EEG data.
        """
        self.project_name = project_name
        self.raw_data = None
        self.preprocessed_data = None
        self.file_path = None
        self.best_channel = None
        
           
    def get_manual_stages(self,stages_path):
        """
        Get a dataframe of manual sleep stages according to csv file.
        The code reads manual stages file according to specific study format, and excludes any event that is not sleep-related.

        Parameters:
        - stages_path (str): Path to stages file

        Returns:
        - stages (pd.dataframe): The manual sleep stages dataframe in the original format except for the replacement of original labels with global numeric labels.
        *** Consider changing to getting a list instead of dataframes with different column names (for each study)
        - annot (mne.Annotation): mne.Annotation object for later eeg epochs extraction
        
        """
        
        if self.project_name == 'NSRR':
            events = pd.read_csv(stages_path, sep = '\t')
            events = events[events['description'].str.match('^Sleep stage.*')==True]
            events['epoch'] = (round(events['onset']/30)).astype(int)
            events = events.reset_index(drop = True)

            # fill missing values
            onset= events['onset'][0]-30 
            onsetVec = []
            while onset > 0 :
                onsetVec.append(onset)
                onset -= 30
            onsetVec.reverse()

            df_to_join = pd.DataFrame({'onset': onsetVec,
                        'duration': 30,
                        'description':'Sleep stage W',
                        'epoch': np.arange(0,len(onsetVec))
                        })
            stages = pd.concat([df_to_join, events], ignore_index = True)
            stages.replace(regex={r'.*W.*':0,r'.*R.*':4,r'.*1.*':1, r'.*2.*':2, r'.*3.*':3,r'.*?.*':0}, inplace=True)
            annot = mne.Annotations(stages['onset'], duration=stages['duration'],
                        description=stages['description'])
            
            
        elif self.project_name == 'NSRR_Stages':
            # events = pd.read_csv(stages_path,usecols=range(3),lineterminator='\n')
            
            # read line by line due to occasional extra commas in the third column
            lines = []
            with open(stages_path, 'r') as csvfile:
                for line in csvfile:
                    lines.append(line.strip())
            header = lines[0].split(',')
            # # The remaining lines are the data
            data = [line.split(',',2) for line in lines[1:]]

            #Create a pandas DataFrame
            events = pd.DataFrame(data, columns=header)
            # include only sleep stages
            pattern = '^.*Stage.*|^.*Wake.*|.*REM.*|.*UnknownStage.*'
            stages = events[events['Event'].str.match(pattern)==True]
            
            # covert duration to float
            def convert_to_integer(s):
                cleaned_string = s.strip()
                return float(cleaned_string)
            
            stages['Duration (seconds)'] = stages['Duration (seconds)'].apply(convert_to_integer)
            stages.loc[:,'Event'].replace(regex={r'.*W.*':0,r'.*R.*':4,r'.*1.*':1, r'.*2.*':2, r'.*3.*':3,r'.*Un.*':0}, inplace=True)
            
            # If stages are not in 30-sec epoch resolution, expand each row to dispaly sleep stage for each epoch
            if (stages['Duration (seconds)'].max() > 30):
                if (stages['Duration (seconds)'].max()/30 < 500):
                    # Create an empty DataFrame for the expanded data
                    expanded_df = pd.DataFrame(columns=stages.columns)

                    # Iterate over each row and expand based on 'Duration'
                    for index, row in stages.iterrows():
                        duration = row['Duration (seconds)']
                        if (duration > 30):
                            # Create duplicate rows based on 'Duration' / 30
                            num_duplicates = int(duration / 30)
                            for i in range(num_duplicates):
                                rowi = row.copy()
                                rowi['Start Time'] = (datetime.strptime(row['Start Time'],"%H:%M:%S")+timedelta(seconds=30)).strftime("%H:%M:%S")
                                rowi['Duration (seconds)'] = 30
                                expanded_df = pd.concat([expanded_df, pd.DataFrame([rowi])], ignore_index=True)
                        else:
                            expanded_df = pd.concat([expanded_df, pd.DataFrame([row])], ignore_index=True)
                    stages = expanded_df
                else:
                    return np.nan
            
            annot = mne.Annotations(np.arange(stages.shape[0]), duration=stages['Duration (seconds)'],
                            description=stages['Event'])
            
        elif self.project_name == 'Dreem':
                        
            stages = pd.read_csv(stages_path)

            # onset_timedeltas = pd.to_timedelta(stages['time'])
            # onset_seconds = onset_timedeltas.dt.total_seconds()
            # Convert stages dataframe to MNE annotations
            annot = mne.Annotations(onset=np.arange(stages.shape[0]),
                                    duration=stages['Duration[s]'].values,
                                    description=stages['Sleep Stage'].values)
        
        return stages, annot
    
    def get_accelerometer(self, file_path, preload):

        if self.project_name == 'Dreem':
        # Load EEG data using MNE-Python
            if self.raw_data is None:
                self.raw_data = mne.io.read_raw_edf(file_path, preload=preload, verbose = 'warning')

            AccChan = ['Respiration y']
            if preload == True:
                self.raw_data.pick(AccChan)

            else:
                print('Wrong project name')
                return np.nan

        return self.raw_data

    def pick_channels(self, file_path, preload):
        """
        Channel selection according to the individual format of channel labels

        Parameters:
        - file_path (str): Path to edf file

        Returns:
        - self.raw_data (mne.Raw): Raw eeg file after channel selection
        
        """
        # Load EEG data using MNE-Python
        if self.raw_data is None:
            self.raw_data = mne.io.read_raw_edf(file_path, preload=preload, verbose = 'warning')
            

        if self.project_name == 'NSRR':
            selectedChannelsPat = re.compile(r'\w{3} \w\w-[M]\w')
            eegChan1 = list(filter(selectedChannelsPat.match,self.raw_data.info['ch_names']))
            eogPat=re.compile(r'EOG')
            eegChan2= list(filter(eogPat.match,self.raw_data.info['ch_names']))
            eegChan= eegChan1 + eegChan2
            if len(eegChan) != 8:
                print('Wrong channels: %s' %self.raw_data.info['ch_names'])
                return np.nan
            if preload == True:
                self.raw_data.pick(eegChan)
            # elif len(eegChan) != 8:
            #     print('Wrong channels: %s' %self.raw_data.info['ch_names'])
            #     return 
                
                
        elif self.project_name == 'NSRR_Stages':

            if any('F3-M2' in s or 'O1-M2' in s for s in self.raw_data.info['ch_names']):
                # STLK
                eegChan = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1', 'E2']
                if preload == True:
                    self.raw_data.pick(eegChan)
                    
            elif any ('F4M1' in s or 'E2M2' in s for s in self.raw_data.info['ch_names']):
                # BOGN
                eegChan = ['F3M2', 'F4M1', 'C3M2', 'C4M1', 'O1M2', 'O2M1', 'E1M2', 'E2M2']
                if preload == True:
                    self.raw_data.pick(eegChan)
                
            elif any('EEG_F3-A2' in s or 'EOG_LOC-A2' in s for s in self.raw_data.info['ch_names']):
                #GSBB, GSDV, GSLH, GSSA, GSSW, MSMI, MSNF, MSQW
                eegChan = ['EEG_F3-A2','EEG_F4-A1','EEG_A1-A2','EEG_C3-A2','EEG_C4-A1','EEG_O1-A2','EEG_O2-A1','EOG_LOC-A2','EOG_ROC-A2']
                if preload == True:
                    self.raw_data.pick(eegChan)
            
            else:
                if 'E1_(LEOG)' in self.raw_data.info['ch_names']:
                    # MSTH
                    eegChan = ['C3','C4','F3','F4','O1','O2','M1','M2','E1_(LEOG)','E2_(REOG)']
                    if preload == True:
                        self.raw_data = mne.set_bipolar_reference(self.raw_data.copy().pick(['F3','F4','C3','C4','O1','O2','E1_(LEOG)','E2_(REOG)','M2','M1']),
                                                  anode = ['F3','C3','O1','F4','C4','O2','E1_(LEOG)','E2_(REOG)'],
                                                  cathode = ['M2','M2','M2','M1','M1','M1','M2','M1'])
                elif 'EOG1' in self.raw_data.info['ch_names']:
                     # MSTR STFN
                    eegChan = ['C3','C4','F3','F4','O1','O2','M1','M2','EOG1','EOG2']
                    if preload == True:
                        self.raw_data = mne.set_bipolar_reference(self.raw_data.copy().pick(['F3','F4','C3','C4','O1','O2','EOG1','EOG2','M2','M1']),
                                                  anode = ['F3','C3','O1','F4','C4','O2','EOG1','EOG2'],
                                                  cathode = ['M2','M2','M2','M1','M1','M1','M2','M1'])
                elif 'L_EOG' in self.raw_data.info['ch_names']:
                    eegChan = ['C3','C4','F3','F4','O1','O2','M1','M2','L_EOG','R_EOG']
                    if preload == True:
                        self.raw_data = mne.set_bipolar_reference(self.raw_data.copy().pick(['F3','F4','C3','C4','O1','O2','L_EOG','R_EOG','M2','M1']),
                                                  anode = ['F3','C3','O1','F4','C4','O2','L_EOG','R_EOG'],
                                                  cathode = ['M2','M2','M2','M1','M1','M1','M2','M1'])
                else:
                    # MSTR STFN
                    eegChan = ['C3','C4','F3','F4','O1','O2','M1','M2','E1','E2']
                    if preload == True:
                        self.raw_data = mne.set_bipolar_reference(self.raw_data.copy().pick(['F3','F4','C3','C4','O1','O2','E1','E2','M2','M1']),
                                                  anode = ['F3','C3','O1','F4','C4','O2','E1','E2'],
                                                  cathode = ['M2','M2','M2','M1','M1','M1','M2','M1'])
#             elif not all([f in self.raw_data.info['ch_names'] for f in eegChan]):
#                 print('Wrong channels')
#                 return np.nan
            
        elif self.project_name == 'Dreem':
            eegChan = ['EEG F7-O1', 'EEG F8-O2','EEG F8-O1', 'EEG F7-O2']
            if preload == True:
                self.raw_data.pick(eegChan)
            elif not all([f in self.raw_data.info['ch_names'] for f in eegChan]):
                print('Wrong channels')
                return np.nan
            
        else:
            print('Wrong project name')
            return np.nan
    
        return self.raw_data
    

    def check_eeg_data(self, file_path, stages_path, preload=True):
        """
        Checks if file can be included for processing based on the following criteria:
        1. More than 2 hours (240 epochs) and less than 12 hours (1440 epochs) of sleep.
        2. The existance of sleep stages Wake, N1, N2 and N3 (REM is alowed to be missing)

        Parameters:
        - file_path (str): The path to the EEG data file.
        - stages_path (str): The path to the manual stages file.
        - preload (bool): Whether to preload the data into memory.
        
        Returns:
        -subject_data_good Y/N indicating if the subject should be excluded from processing
        
        """
        subject_data_good='Y'

        
        
        self.raw_data = self.pick_channels(file_path, preload = False)
        if type(self.raw_data) != mne.io.edf.edf.RawEDF:
            print('Failed to pick channels')
            subject_data_good = 'N'
            return subject_data_good
       
        
        try: 
            # print('try annot')
            _, annot = self.get_manual_stages(stages_path)
            # annot.description.tolist()
            # self.epochs, self.raw_data, self.croppeData = self.create_epochs(stages_path)
            unqstages = np.unique([int(i) for i in annot.description.tolist()]).tolist()
            vals = [0,1,2,3]
            # print('found annot')
            
        except Exception as e:
            print('Failed to extract epochs', e)
            subject_data_good = 'N'
            return subject_data_good
            
        if ((annot.description.shape[0] < 1440) & (annot.description.shape[0] > 240) & all(v in unqstages for v in vals)) :
            subject_data_good='Y'
        else:
            subject_data_good='N'
            print('Bad file: %s' %file_path)
            print('Length (epochs): %d' %annot.description.shape[0])
            print('Unique stages: %s' %unqstages)
            
        return subject_data_good
    
    def check_data_for_cm(self, noisePath, StagesPath, noisethreshold = 0.25):
        
        """
        Checks if file can be included in confusion Matrix:
        1. More than 2 hours (240 epochs) and less than 12 hours (1440 epochs) of sleep.
        2. The existance of sleep stages Wake, N1, N2 and N3 (REM is alowed to be missing)
        3. less than X (noisethreshold) percent of noise 

        Parameters:
        - noise_path (str): The path to the noise file.
        - stages_path (str): The path to the manual stages file.
        - noisethreshold (float): desire proportion of noise to be excluded.
        
        Returns:
        -subject_data_good Y/N indicating if the subject should be excluded from confusion matrix
        
        """
        subject_data_good = 'Y'
        
        # load noise data
        noise = pd.read_csv(noisePath, index_col=0)
        # calculate the noise proportion of the channel with the minimal noise
        noisecount = (noise>0).sum().min()
        noiseprob = noisecount/noise.shape[0]
        print(noiseprob)

        if noiseprob > noisethreshold:
            subject_data_good = 'N'
            print('Excluded due to %d noise proportion' %noiseprob)
            return subject_data_good
        
        _, annot = self.get_manual_stages(StagesPath)
        unqstages = np.unique([int(i) for i in annot.description.tolist()]).tolist()
        vals = [0,1,2,3]
        if ((annot.description.shape[0] < 1440) & (annot.description.shape[0] > 240) & all(v in unqstages for v in vals)) :
            subject_data_good='Y'
        else:
            subject_data_good='N'
            print('Bad file: %s' %noisePath)
            print('Length (epochs): %d' %annot.description.shape[0])
            print('Unique stages: %s' %unqstages)
        
        return subject_data_good
        
    def create_epochs(self, stages_path, preload = True):
        """
        Extract epochs from raw eeg based on manual timing
        *** Requires data to be loaded. 

        Parameters:
        - stages_path (str): The path to the manual stages file.
        - preload (bool): Whether to preload the data into memory.
        
        Returns:
        - self.epochs (mne.Epochs)
        - self.croppedData (mne.Raw): relevant only in cases where the raw data has to be cropped to alin epochs with manual timing
        
        """
        
        dataMan, annot = self.get_manual_stages(stages_path)
        
        if self.project_name == 'NSRR':
            self.croppeData = self.raw_data.copy().crop(tmin = dataMan.loc[0,'onset'],
                                  tmax = dataMan.onset.values[-1])
        else:
            self.croppeData = self.raw_data
                        
        # Mutual steps regardless of project name:
        # create annotations
        
        self.raw_data.set_annotations(annot, emit_warning = False)
        annotation_desc_2_event_id = {
                        "0": 0,
                        "1": 1,
                        "2": 2,
                        "3": 3,
                        "4": 4}

        events_from_annot, _ = mne.events_from_annotations(
            self.raw_data, event_id = annotation_desc_2_event_id)

        tmax = 30.0 - 1.0 / self.raw_data.info["sfreq"]
        self.epochs = mne.Epochs(raw = self.raw_data,
                            events = events_from_annot,
                            tmin = 0,
                            tmax = tmax,
                            baseline = None,
                            verbose = 'warning')
        
        return self.epochs, self.croppeData

    
    def preprocess_eeg_data(self, file_path, stages_path, preload=True, l_freq=0.40, h_freq=30):
        
        """
        Preprocess EEG data using MNE-Python.

        Parameters:
        - file_path (str): The path to the EEG data file.
        - stages_path (str): Path to stages file
        - preload (bool): Whether to preload the data into memory.
        - l_freq/h_freq (int, optional): lower and upper pass-band edges.

        Returns:
        - self.epochs (mne.epochs): The preprocessed EEG data.
        - self.croppedData (mne.Raw): relevant only in cases where the raw data has to be cropped to align epochs with manual timing not relevent for Dreem files with no manual scoring

        """
        # Load EEG data using MNE-Python and pick channels
        self.raw_data = mne.io.read_raw_edf(file_path, preload=preload, verbose='warning')
        
        self.raw_data = self.pick_channels(file_path, preload=True)
        self.preprocessed_data = self.raw_data.filter(l_freq=l_freq, h_freq=h_freq, verbose='warning')

        if self.project_name == 'NSRR':
            
            self.epochs, self.croppedData = self.create_epochs(stages_path)
            
            return self.epochs, self.croppedData
        # self.create_epochs not relevent for Dreem because theres no manual scoring

        elif self.project_name == 'Dreem':
            
            #epoch Dreem data
            
            stages = pd.read_csv(stages_path)
            # onset_timedeltas = pd.to_timedelta(stages['time'])
            # onset_seconds = onset_timedeltas.dt.total_seconds()
            # Epoch the data into 30-second windows
            self.events = mne.make_fixed_length_events(self.preprocessed_data, start=0, duration=30)
            # Epoch the EEG data
            self.epochs = mne.Epochs(self.preprocessed_data, self.events, tmin=0, tmax=30, baseline=None)
            # Convert stages dataframe to MNE annotations
            annot = mne.Annotations(onset=np.arange(stages.shape[0]),
                                    duration=stages['Duration[s]'].values,
                                    description=stages['Sleep Stage'].values)
            # Add annotations to the epochs
            self.preprocessed_data.set_annotations(annot)
            self.croppedData = self.raw_data
            
            return self.epochs,self.croppedData

    def compute_Alg_stages(self,data):
        df_pred = pd.DataFrame()
        
        if self.project_name == 'Dreem':
            
            ch_names = data.info['ch_names'] 
        else:
            ch_names = data.info['ch_names']
        for i in ch_names:
            sls = yasa.SleepStaging(data, eeg_name = i)
            y_pred = sls.predict()
            confidence = sls.predict_proba().max(1)

            df_pred[i+'_Y'] = y_pred
            df_pred[i+'_Y_conf'] = sls.predict_proba().max(1).tolist()

        df_pred.replace(regex={r'.*W.*':0,r'.*R.*':1,r'.*N1.*':2, r'.*N2.*':3, r'.*N3.*':4,r'.*?.*':np.nan}, inplace=True)
        
        conf_columns = df_pred.filter(like='_Y_conf').columns
        df_pred['Highest_Conf_Score_Column'] = df_pred[conf_columns].idxmax(axis=1)
        df_pred['Highest_Conf_Score_Channel'] = df_pred['Highest_Conf_Score_Column'].apply(lambda x: x.replace('_Y_conf', ''))

        df_pred['Highest_Conf_Score_Val'] = df_pred.apply(
            lambda row: row[f"{row['Highest_Conf_Score_Channel']}_Y"], axis=1
        )

        return df_pred['Highest_Conf_Score_Val']

        # api = USleepAPI(api_token =token)
        # # api.delete_all_sessions()
        # hypnogram, _ = api.quick_predict(
        #     input_file_path=file,
        #     anonymize_before_upload=False
        # )
        # df_pred['U'] = hypnogram['hypnogram']
        # df_pred.U.replace([4,2],[2,4], inplace = True)

        if (self.project_name == 'NSRR') | (self.project_name == 'NSRR_Stages'):
            ch_names = data.info['ch_names'][:-2]
        else:
            ch_names = data.info['ch_names']
        for i in ch_names:
            sls = yasa.SleepStaging(data, eeg_name = i)
            y_pred = sls.predict()
            confidence = sls.predict_proba().max(1)

            df_pred[i+'_Y'] = y_pred
            df_pred[i+'_Y_conf'] = sls.predict_proba().max(1).tolist()

        # df_pred.replace(regex={r'.*W.*':0,r'.*R.*':1,r'.*N1.*':2, r'.*N2.*':3, r'.*N3.*':4,r'.*?.*':9}, inplace=True)
        
        # api = USleepAPI(api_token =token)
        # # api.delete_all_sessions()
        # hypnogram, _ = api.quick_predict(
        #     input_file_path=file,
        #     anonymize_before_upload=False
        # )
        # df_pred['U'] = hypnogram['hypnogram']
        # df_pred.U.replace([4,2],[2,4], inplace = True)
        return df_pred

        # df_pred.to_csv(f"/mnt/home/geylon/ceph/data/NSRR/Results/SleepStagesComb2/{fileName.replace('.edf','')}.csv")
        # print('saved file '+file)
    
    def compute_fft(self, preprocessed_data, info):
        """
        Calculate fft.

        Parameters:
        - preprocessed_data (mne.epochs): The preprocessed EEG data.
        - info data structure (mne.Info)

        Returns:
        - fft (numpy.ndarray) of shape (n_epochs, n_channels, n_times)
        
        """
        
        if (self.project_name == 'NSRR') | (self.project_name == 'NSRR_Stages'):
            ch_names = preprocessed_data.info['ch_names'][:-2]
        else:
            ch_names = preprocessed_data.info['ch_names']
        preprocessed_data = preprocessed_data.get_data(units = 'uV', picks = ch_names)
 
            
        ### FFT
        # define parameters for fft
        winlen = info['sfreq'] * 4
        overlap = info['sfreq'] * 2
        hzL = np.linspace(0, info['sfreq'] / 2, int(winlen / 2) + 1)
        # initilize the fft dataset to the same size as the data, data but only up to the nyquist frequency
        fftWelch = np.zeros((preprocessed_data.shape[0],preprocessed_data.shape[1],len(hzL)))

        # compute fft by iterating through channels and epochs
        for eleci in range(preprocessed_data.shape[1]):
            for epochi in range(preprocessed_data.shape[0]):
                epochData = preprocessed_data[epochi,eleci,:]
                numFrames = int((len(epochData) - winlen) / overlap) + 1

                # perform spectral analysis using the Welch method (divide epoch into overlapping segments and then average the periodogram)
                fftA = np.zeros(len(hzL))
                for j in range(numFrames):
                    frameData = epochData[int(j * overlap) :int( j * overlap + winlen)]
                    fftTemp = np.fft.fft(np.hamming(winlen) * frameData) / winlen
                    fftA += np.square(2 * np.abs(fftTemp[:len(hzL)]))

                fftWelch[epochi,eleci :] = fftA / numFrames
            
        return fftWelch
        
    
    def compute_noise_matrices(self, preprocessed_data, info, sd_crt = 2):
        # insert code to compute all the noise matrices
        
        if (self.project_name == 'NSRR') | (self.project_name == 'NSRR_Stages'):
            ch_names = preprocessed_data.info['ch_names'][:-2]
        else:
            ch_names = preprocessed_data.info['ch_names']
        preprocessed_data = preprocessed_data.get_data(units = 'uV', picks = ch_names)
        
        
        maxk_amp_all = np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
        if True:
            # Strat by calculating the maxk caritia for interpolation and intial calculation of fft
            abs_data = np.abs(preprocessed_data)
            amp_threshold = 200  # micro volts
            min_amp_count = 10*info['sfreq']
            for eleci in range(abs_data.shape[1]):
                indices = np.argsort(abs_data, axis=2)[:, eleci, -10:]

                for epochi in range(abs_data.shape[0]):
                    if np.sum(abs_data[epochi,eleci,indices[epochi]] > amp_threshold) >= min_amp_count:  
                        maxk_amp_all[epochi, eleci] = 1;
                        timepoints =  np.where(preprocessed_data[epochi,eleci,:] > amp_threshold)
                        preprocessed_data[epochi,eleci,timepoints]  = np.nan

                    else:
                        maxk_amp_all[epochi, eleci] = 0;
                        # replace noise w/ nan
                        timepoints =  np.where(abs_data[epochi,eleci,:] > amp_threshold)
                        preprocessed_data[epochi,eleci,timepoints]  = np.nan
                        # replace short length noisy data w/ interpolation
                        epoch_data = preprocessed_data[epochi]  # Get data for the current epoch
                        nan_indices = np.isnan(epoch_data)
                        # Interpolate the NaN values using the non-NaN values
                        interpolated_values = np.interp(np.flatnonzero(nan_indices), np.flatnonzero(~nan_indices), epoch_data[~nan_indices])
                        # Assign the interpolated values back to the data array
                        epoch_data[nan_indices] = interpolated_values
                        # Update the data array with the interpolated values for the current epoch
                        preprocessed_data[epochi] = epoch_data   
           
            
            ### FFT
            # define parameters for fft
            winlen = info['sfreq'] * 4
            overlap = info['sfreq'] * 2
            hzL = np.linspace(0, info['sfreq'] / 2, int(winlen / 2) + 1)
            # initilize the fft dataset to the same size as the data, data but only up to the nyquist frequency
            fftWelch = np.zeros((preprocessed_data.shape[0],preprocessed_data.shape[1],len(hzL)))
            
            # compute fft by iterating through channels and epochs
            for eleci in range(preprocessed_data.shape[1]):
                for epochi in range(preprocessed_data.shape[0]):
                    epochData = preprocessed_data[epochi,eleci,:]
                    numFrames = int((len(epochData) - winlen) / overlap) + 1
                    
                    # perform spectral analysis using the Welch method (divide epoch into overlapping segments and then average the periodogram)
                    fftA = np.zeros(len(hzL))
                    for j in range(numFrames):
                        frameData = epochData[int(j * overlap) :int( j * overlap + winlen)]
                        fftTemp = np.fft.fft(np.hamming(winlen) * frameData) / winlen
                        fftA += 2 * np.abs(fftTemp[:len(hzL)])

                    fftWelch[epochi,eleci :] = fftA / numFrames
            
            #sum up frequancies in the 10 to 20 Hz range
            freq_res = 0.25
            delta = [0.75, 4] 
            theta = [4, 8]
            alpha = [8, 12]
            beta = [12, 20]

            winlen = info['sfreq'] * 4
            overlap = info['sfreq'] * 2

            hzL = np.linspace(0, info['sfreq'] / 2, int(winlen / 2) + 1)  # frequencies for every window

            # get indices of each frequency band
            deltaidx = np.searchsorted(hzL, delta)
            thetaidx = np.searchsorted(hzL, theta)
            alphaidx = np.searchsorted(hzL, alpha)
            betaidx = np.searchsorted(hzL, beta)
            
            fft_delta = np.squeeze(np.sum(fftWelch[:, :,deltaidx[0]:deltaidx[-1] + 1],axis=2))
            fft_theta = np.squeeze(np.sum(fftWelch[:, :,thetaidx[0]:thetaidx[-1] + 1],axis=2))
            fft_alpha = np.squeeze(np.sum(fftWelch[:, :,alphaidx[0]:alphaidx[-1] + 1],axis=2))
            fft_beta = np.squeeze(np.sum(fftWelch[:, :,betaidx[0]:betaidx[-1] + 1],axis=2))
            
            
            ### Additional noise parameters
            
            var_epoch_all = np.var(preprocessed_data, axis=2)  # variance
            kurt_epoch_all = kurtosis(preprocessed_data, axis=2)  # kurtosis
            mobility_epoch_all = np.sqrt(np.var(np.diff(preprocessed_data, axis=2), axis=2)) / (var_epoch_all)
            complexity_epoch_all = np.sqrt(np.var(np.diff(np.diff(preprocessed_data, axis=2), axis=2), axis=2)) / np.var(np.diff(preprocessed_data, axis=2), axis=2)
            max_amp = np.max(np.abs(preprocessed_data), axis=2)
            diff_all = np.median(np.diff(preprocessed_data, axis=2), axis=2)
            
            # get stds and medians
            median_var_all = np.median(var_epoch_all,axis = 0);
            sd_var_all = np.std(var_epoch_all,axis = 0);

            median_kurt_all = np.median(kurt_epoch_all,axis = 0);
            sd_kurt_all = np.std(kurt_epoch_all,axis=0);

            median_delta_all = np.median(fft_delta, axis = 0);
            sd_delta_all = np.std(fft_delta, axis = 0);
            
            median_theta_all = np.median(fft_theta, axis = 0);
            sd_theta_all = np.std(fft_theta, axis = 0);
            
            median_alpha_all = np.median(fft_alpha, axis = 0);
            sd_alpha_all = np.std(fft_alpha, axis = 0);
            
            median_beta_all = np.median(fft_beta, axis = 0);
            sd_beta_all = np.std(fft_beta, axis = 0);

            median_amp_all = np.median(max_amp, axis=0);
            sd_amp_all = np.std(max_amp, axis=0);

            median_mob_all = np.median(mobility_epoch_all, axis=0);
            sd_mob_all = np.std(mobility_epoch_all, axis=0);

            median_comp_all = np.median(complexity_epoch_all, axis=0);
            sd_comp_all = np.std(complexity_epoch_all, axis=0);

            median_diff_all = np.median(diff_all, axis=0);
            std_diff_all = np.std(diff_all,axis=0);
            
            #Initialize empty datasets for all noise features
            delta_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            theta_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            alpha_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            beta_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            
            kurt_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            var_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            mob_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            comp_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            amp_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            diff_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))

            
            # mark noise based on X (X=sd_crt) sd deviation from median
            
            for eleci in range(median_var_all.shape[0]):
                delta_mark_all[:, eleci] = np.where(np.abs(fft_delta[:,eleci]) > sd_crt * sd_delta_all[eleci] + median_delta_all[eleci], 1, 0)
                theta_mark_all[:, eleci] = np.where(np.abs(fft_theta[:,eleci]) > sd_crt * sd_theta_all[eleci] + median_theta_all[eleci], 1, 0)
                alpha_mark_all[:, eleci] = np.where(np.abs(fft_alpha[:,eleci]) > sd_crt * sd_alpha_all[eleci] + median_alpha_all[eleci], 1, 0)
                beta_mark_all[:, eleci] = np.where(np.abs(fft_beta[:,eleci]) > sd_crt * sd_beta_all[eleci] + median_beta_all[eleci], 1, 0)            
                kurt_mark_all[:, eleci] = np.where(np.abs(kurt_epoch_all[:,eleci]) > sd_crt * sd_kurt_all[eleci] + median_kurt_all[eleci], 1, 0)
                #var_mark_all[:, eleci] = np.where(var_epoch_all[:,eleci] > sd_crt * sd_var_all[eleci] + median_var_all[eleci], 1, 0)
                var_mark_all[:, eleci] = np.where((np.abs(var_epoch_all[:,eleci]) > sd_crt * sd_var_all[eleci] + median_var_all[eleci]) |
                                                 (var_epoch_all[:,eleci] < 1),
                                                  1, 0)
                # add precentile for 2% leass
                
                mob_mark_all[:, eleci] = np.where(np.abs(mobility_epoch_all[:,eleci]) > sd_crt * sd_mob_all[eleci] + median_mob_all[eleci], 1, 0)
                comp_mark_all[:, eleci] = np.where(np.abs(complexity_epoch_all[:,eleci]) > sd_crt * sd_comp_all[eleci] + median_comp_all[eleci], 1, 0)
                amp_mark_all[:, eleci] = np.where(
                    
                    np.abs(max_amp[:,eleci]) > sd_crt * sd_amp_all[eleci] + median_amp_all[eleci], 1, 0)
                diff_mark_all[:, eleci] = np.where(np.abs(diff_all[:,eleci]) > sd_crt * std_diff_all[eleci] + median_diff_all[eleci], 1, 0)
                


                    
                    
        return delta_mark_all, theta_mark_all, alpha_mark_all, beta_mark_all, kurt_mark_all, var_mark_all, mob_mark_all, comp_mark_all, amp_mark_all, diff_mark_all, maxk_amp_all, fftWelch, ch_names
    
    def noise_summary(self,delta_mark_all, theta_mark_all, alpha_mark_all, beta_mark_all, kurt_mark_all, var_mark_all, mob_mark_all, comp_mark_all, amp_mark_all, diff_mark_all, maxk_amp_all, ch_names):
        """
        Create a NumPy array that quantifies the amount of noise in each channel instead of choosing best electrode
        ***applies to NSRR data!***
        """
        
        # # Concatenate arrays
        mark_all = np.concatenate((
                                   alpha_mark_all[:, :, np.newaxis], beta_mark_all[:, :, np.newaxis],
                                    theta_mark_all[:, :, np.newaxis], delta_mark_all[:, :, np.newaxis],
                                   kurt_mark_all[:, :, np.newaxis],var_mark_all[:, :, np.newaxis],
                                   mob_mark_all[:, :, np.newaxis], comp_mark_all[:, :, np.newaxis],
                                   amp_mark_all[:, :, np.newaxis],diff_mark_all[:, :, np.newaxis],
                                   maxk_amp_all[:, :, np.newaxis]),  axis=2)
        
        mark_all_by_elec = np.any(mark_all, axis=2)
        
        # Sum ones
        mark_all_by_elec_sum = np.sum(mark_all_by_elec, axis=0)
        # set pandas df with channel name and noise count
        
        elecdf = pd.DataFrame({'name':ch_names, 'n_noise':mark_all_by_elec_sum})
        
        mark_prob = (np.sum(mark_all, axis=2)/np.shape(mark_all)[2])
        df_pred = pd.DataFrame()
        for i in range(mark_all_by_elec.shape[1]):
            df_pred[f"{elecdf.iloc[i]['name']} - noise"] = (mark_prob[:,i])
        
        return elecdf, df_pred, mark_all

    def choose_best_electrode(self, info, mark_all):
        
        if self.project_name == 'Dreem':

            """
            Choose the best electrode based on multiple criteria.

            Parameters:
            - epochs (mne.Epochs): The EEG epochs.
            - fft_mark_all, kurt_mark_all, var_mark_all, mob_mark_all, comp_mark_all,
              amp_mark_all, diff_mark_all, maxk_amp_all (numpy.ndarray): Mark arrays.

            Returns:
            - best_channel (numpy.ndarray): Array of indices indicating the best channel for each epoch.
            - rejected_epochs (numpy.ndarray): Array indicating rejected epochs.
            - final_mark (numpy.ndarray): Final mark array.
            """
           

            # Find the number of channels and epochs
            num_epochs, num_channels, _ = mark_all.shape

            # Create an array to store the sum of ones for each channel
            noise_per_channel = np.sum(mark_all, axis=2)

            # Find the channel with the minimum number of ones (least noise)
            self.best_channel = np.argmin(noise_per_channel, axis=1)

            # Flag noisy epochs for the "best channel" for that epoch
            best_channel_mark = noise_per_channel[np.arange(num_epochs), self.best_channel] > 0

            # Create final_mark as a binary vector
            final_mark = best_channel_mark

            preprocessed_data_reject = np.logical_not(~final_mark)

            rejected_epochs = preprocessed_data_reject
            

        return self.best_channel, rejected_epochs, final_mark
    
    
    def SoWu_algo(self, info, fftWelch, rejected_epochs):

        # define frequency bands
        freq_res = 0.25
        delta = [0.75, 4] 
        theta = [4, 8]
        alpha = [8, 12]
        beta = [12, 20]
        allfreq = [0, 20]

        winlen = info['sfreq'] * 4
        overlap = info['sfreq'] * 2

        hzL = np.linspace(0, info['sfreq'] / 2, int(winlen / 2) + 1)  # frequencies for every window
        
        # get indices of each frequency band
        deltaidx = np.searchsorted(hzL, delta)
        thetaidx = np.searchsorted(hzL, theta)
        alphaidx = np.searchsorted(hzL, alpha)
        betaidx = np.searchsorted(hzL, beta)
        allfreqidx = np.searchsorted(hzL, allfreq)

        fftWelch = pd.DataFrame(fftWelch)
        
        # replace values of noisy epochs with nan
        fftWelch[rejected_epochs] = np.nan
        
        # get power of relevant frequencies
        power_all = fftWelch.iloc[:,allfreqidx[0]:allfreqidx[1]+1]
        
        ### within each frequency window, replace values that are greater by more 3std from the mean with nan
        std_threshold = 3
        std = np.std(power_all, axis=0)
        power_all[np.abs(power_all - np.mean(power_all)) > std_threshold * std] = np.nan

        power_filt = power_all.copy()
        power_nan = np.isnan(power_filt)

        #intepolate missing values across epochs, per each frequeny window (columns)
        for i in power_filt.columns:
            nan = power_nan.loc[:, i]
            if nan.all():  # Check if the whole column is NaN
                power_filt.loc[:, i] = 0    

            else:
                power_filt.loc[nan, i] = np.interp(np.flatnonzero(nan), np.flatnonzero(~nan), power_filt.loc[~nan, i])

        if self.project_name == 'Dreem':
            # Create an empty list to hold the indices of noisy episodes
            noisy_episodes = []

            # Define the length of an episode and noise threshold
            episode_length = 40
            noise_threshold = 0.6

            # Iterate through the epochs to find noisy episodes
            for start_idx in range(0, len(power_nan) - episode_length + 1,episode_length):
                end_idx = start_idx + episode_length
                episode = rejected_epochs[start_idx:end_idx]

                # Calculate the proportion of noisy epochs in the episode
                noise_proportion = (episode.sum()/ episode_length)

                # If the proportion of noisy epochs is greater than the threshold, store the indices
                if noise_proportion > noise_threshold:
                    noisy_episodes.extend(range(start_idx, end_idx))

            # Convert the list to a NumPy array for easier manipulation later
            noisy_episodes = np.array(noisy_episodes)   

            # reassign 0 values to episodes with more than 75% noise
            if len(noisy_episodes) > 0 :
                power_filt.iloc[noisy_episodes] = 0
                
        # integrate values (after interpolation) to get power of full frequency bands
        delta_power = simps(power_filt.iloc[:,deltaidx[0]:deltaidx[1]], dx=freq_res)
        delta_smoothed = gaussian_filter(delta_power, sigma=15, mode='wrap')
        
        alpha_power = simps(power_filt.iloc[:,alphaidx[0]:alphaidx[1]], dx=freq_res)
        theta_power = simps(power_filt.iloc[:,thetaidx[0]:thetaidx[1]], dx=freq_res)
        total_power = simps(power_filt, dx=freq_res)
        
        # calculate alpha to theta+delta ratio
        ratioAT = alpha_power/ (theta_power + delta_power)
        ratioAT_smoothed = gaussian_filter(ratioAT, sigma=15, mode = 'wrap')

        ### find Delta peaks from intial mnima to avoid delta noise at sleep intiation         
        # Define a window size (+/- 45 minutes)
        # Calculate the local minima before and after each peak
        
        local_minima = []
        window_size = 90
        
        if self.project_name == 'Dreem':
            # get median of delta power
            minAmp = np.median(delta_smoothed)
            # get 5th and 95th perentiles for prominence calculation
            perc = np.percentile(delta_smoothed, [5, 95])
            perc_75 = np.percentile(delta_smoothed,75)
            # Find the first index where the signal is below the 75th percentile
            start_index = next((i for i, v in enumerate(delta_smoothed) if v <= perc_75), 0)
            prom = (perc[1]-perc[0])/3
            peaks, _ = find_peaks(delta_smoothed[start_index:], width = 20, height = minAmp,  distance = 60)#,prominence = prom)
            
            peaks = peaks + start_index # Adjusting the indices of the peaks
            ratio_peaks, _ = find_peaks(ratioAT_smoothed, height = np.percentile(ratioAT_smoothed,95))

            # peaks, _ = find_peaks(delta_smoothed[np.argmin(delta_smoothed[:100]):],width=20, height=(50,550), prominence=30, distance = 60)
            # peaks = peaks + np.argmin(delta_smoothed[:100])
            
            for peak in peaks:
                # Get only non-noisy epochs in the window before and after the peak
                before_min_range = [i for i in range(max(1, peak - window_size), peak) if i not in noisy_episodes]
                after_min_range = [i for i in range(peak + 1, min(len(delta_smoothed) - 1, peak + window_size)) if i not in noisy_episodes]
                
                # Local minima before the peak
                before_min = before_min_range[np.argmin(delta_smoothed[before_min_range])]
                local_minima.append(before_min)

                # Local minima after the peak
                after_min = after_min_range[np.argmin(delta_smoothed[after_min_range])]
                local_minima.append(after_min)
        
            window_size = 60
            local_minimaAT = []   
            for peak in ratio_peaks:
                # Get only non-noisy epochs in the window before and after the peak
                before_min_range = [i for i in range(max(1, peak - window_size), peak) if i not in noisy_episodes]
                after_min_range = [i for i in range(peak + 1, min(len(ratioAT_smoothed) - 1, peak + window_size)) if i not in noisy_episodes]

                # Local minima before the peak
                before_min = before_min_range[np.argmin(ratioAT_smoothed[before_min_range])]
                local_minimaAT.append(before_min)

                # Local minima after the peak
                after_min = after_min_range[np.argmin(ratioAT_smoothed[after_min_range])]
                local_minimaAT.append(after_min)
                
        if self.project_name == 'NSRR':
            
            
            nanind = np.nonzero(rejected_epochs)[0].tolist()
            try:
                ind_g_100 = next(x for x, val in enumerate(nanind) if val >= 100)
                if ind_g_100-1 > 60:
                    strt = nanind[ind_g_100-1]+1
                else:
                    strt = 0
            except:
                strt = 0
                
            try:
                ind_g_100_back = next(x for x, val in enumerate(nanind) if val > rejected_epochs.shape[0]-100)
                if len(nanind)-(ind_g_100_back+1) > 60:
                    endind = nanind[ind_g_100_back+1]-1
                else:
                    endind = rejected_epochs.shape[0]
            except:
                endind = rejected_epochs.shape[0]

            # get median of delta power
            minAmp = np.median(delta_smoothed)
            # get 5th and 95th perentiles for prominence calculation
            perc = np.percentile(delta_smoothed, [5, 95])
            prom = (perc[1]-perc[0])/5
            # if False:
            peaks, _ = find_peaks(delta_smoothed[strt:endind], width = 20, height = minAmp, prominence = prom, distance = 60)
            ratio_peaks, _ = find_peaks(ratioAT_smoothed[strt:endind], height = np.percentile(ratioAT_smoothed,95))
            peaks = peaks + strt
            ratio_peaks = ratio_peaks + strt
            print(str(strt))
            print(str(endind))
            print(peaks)
            print(rejected_epochs.shape[0])
        
            for peak in peaks:
                # Local minima before the peak
                # for future develpment: add a slope value to validate the local minima
                before_min = max(strt, peak - window_size)
                local_minima.append(np.argmin(delta_smoothed[before_min:peak]) + before_min)

                # Local minima after the peak
                after_min = min(endind, peak + window_size)
                local_minima.append(np.argmin(delta_smoothed[peak:after_min]) + peak)
        
            # find local mnima for ratio_peaks
            window_size = 60
            local_minimaAT = []   
            for peak in ratio_peaks:
                before_min = max(strt, peak - window_size)
                local_minimaAT.append(np.argmin(ratioAT_smoothed[before_min:peak])+before_min)
                
                after_min = min(endind, peak + window_size)
                local_minimaAT.append(np.argmin(ratioAT_smoothed[peak:after_min]) + peak)
                # Get only non-noisy epochs in the window before and after the peak
#                 before_min_range = [i for i in range(max(1, peak - window_size), peak)]
#                 after_min_range = [i for i in range(peak + 1, min(len(ratioAT_smoothed) - 1, peak + window_size))]

#                 # Local minima before the peak
#                 before_min = before_min_range[np.argmin(ratioAT_smoothed[before_min_range])]
#                 local_minimaAT.append(before_min)

#                 # Local minima after the peak
#                 after_min = after_min_range[np.argmin(ratioAT_smoothed[after_min_range])]
#                 local_minimaAT.append(after_min)
                
                
        #define window as 10 minutes after local minima from first SWA cycle
        
        clean_ref_window = []
        i = local_minima[0]
        while len(clean_ref_window) < 20 and i < len(ratioAT):
            if not rejected_epochs[i]:
                clean_ref_window.append(i)
            i += 1
        print(f'clean_ref_window:{clean_ref_window}')
        # find sleep_onet based on alpha/theta ration

        # Initialize variables for sleep-onset
        so_algo = None
        consecutive_count_so = 0
        target_value = (np.mean(ratioAT[clean_ref_window]) + np.std(ratioAT[clean_ref_window]))
        
        i = local_minima[0]-1
        while consecutive_count_so < 10 and i >=0:
            if not rejected_epochs[i]:
                epoch = ratioAT[i]
                if epoch >= target_value:
                    consecutive_count_so +=1
                    if consecutive_count_so == 10:
                        so_algo = i+9
                        break
                else:
                    consecutive_count_so = 0
            i -= 1
        
        if (so_algo==0) | (so_algo is None):
            print('failed loop so')
            
            try:
                so_algo = np.where(ratioAT[:local_minima[0]] >= (np.nanmean(clean_ref_window) + np.nanstd(clean_ref_window)))[0][-1]
                print('logic')
            except:  # if there's no epoch that answers this criteria use local mnima as SO
                so_algo = np.argmin(ratioAT[:local_minima[0]])+1
                print('local min')
                
        if so_algo <= max(0,local_minima[0] - 40):
            so_algo = local_minima[0]
                
        # find Final awaknning based on alpha/theta ratio 
        
        consecutive_count = 0  # Variable to count consecutive epochs meeting the condition        
        # Case of significant ⍺/(θ+δ) power towrds the end of the recording
        # Treat the local mnima befor the ⍺ peak as the end of the recording
        if len(local_minimaAT) > 1 and (ratio_peaks[-1] > peaks[-1]):
            end_wu = local_minimaAT[-2]
        else:
            end_wu = len(ratioAT) -1
            
        # Initialize wu_algo to None. Will update this if we find a match.
        wu_algo = None

        # Loop through the array in reverse, starting from the end
        for i in range(end_wu, local_minima[-2], -1):
            epoch = ratioAT[i]

            # Check if the epoch meets the condition
            if epoch <= target_value:
                # delete next row 16.10 16:25
                if not rejected_epochs[i]:
                    consecutive_count += 1  # Increment the count
                    if consecutive_count == 10:
                        wu_algo = i+9  # Found the start of the 10 consecutive epochs
                        break  # Exit the loop
                        print(wu_algo)
            else:
                consecutive_count = 0  # Reset the count

        if (wu_algo==0) | (wu_algo is None):
            print('failed loop wu')
            try:
                wu_algo = local_minima[-2] +np.where(ratioAT[local_minima[-2]:end_wu] <= target_value_wu[0][-1])

            except:  # if there's no epoch that answers this criteria
                wu_algo = local_minima[-2] + np.argmin(ratioAT[local_minima[-2]:end_wu])-1
                print([so_algo, wu_algo])
                
                                                     
                
        return so_algo, wu_algo, delta_smoothed, ratioAT, alpha_power, delta_power, theta_power, peaks
        # return so_algo, wu_algo, delta_smoothed, ratioAT, alpha_power, delta_power, theta_power, peaks, local_minima, local_minimaAT, ratio_peaks, ratioAT_smoothed

        
    def find_position(self, numbers):
        
        consecutive_count = 0
        consecutive_position = None

        for i, num in enumerate(numbers):
            if num < 8:
                consecutive_count += 1
                if consecutive_count == 1:
                    consecutive_position = i
                if consecutive_count == 5:
                    return consecutive_position*10
            else:
                consecutive_count = 0
                consecutive_position = None

        # Check for the case where there are two consecutive numbers
        # meeting the criteria from the fourth position.
        for i in range(4, len(numbers)):
            if numbers[i] < 8 and numbers[i - 1] < 8:
                return (i - 1)*10
            

        return None
     

    def SoWu_algo_check_conditions(self, info, fftClean, rejected_epochs, nan_flag, prom_flag):
        
        # define frequency bands
        freq_res = 0.25
        delta = [0.75, 4] 
        theta = [4, 8]
        alpha = [8, 12]
        beta = [12, 20]
        allfreq = [0, 20]

        winlen = info['sfreq'] * 4
        overlap = info['sfreq'] * 2

        hzL = np.linspace(0, info['sfreq'] / 2, int(winlen / 2) + 1)  # frequencies for every window
        
        # get indices of each frequency band
        deltaidx = np.searchsorted(hzL, delta)
        thetaidx = np.searchsorted(hzL, theta)
        alphaidx = np.searchsorted(hzL, alpha)
        betaidx = np.searchsorted(hzL, beta)
        allfreqidx = np.searchsorted(hzL, allfreq)

        # convert fft data to pandas dataframe
        fftClean = pd.DataFrame(fftClean)
        # replace values of noisy epochs with nan - Imlement noise detection results
        fftClean[rejected_epochs] = np.nan
        
        # get power of relevant frequencies
        power_all = fftClean.iloc[:, allfreqidx[0]:allfreqidx[1] + 1].copy()

        # Threshold for standard deviation
        std_threshold = 3

        # Calculate standard deviation and mean
        std = np.std(power_all, axis=0)
        mean_power_all = np.mean(power_all)

        # Replace values that are greater by more than std_threshold * std from the mean with NaN
        mask = np.abs(power_all - mean_power_all) > std_threshold * std
        power_all[mask] = np.nan

        power_filt = power_all.copy()
        power_nan = np.isnan(power_filt)

        #intepolate missing values across columns (frequencies)
        for i in power_filt.columns:
            nan = power_nan.loc[:, i]
            if nan.all():  # Check if the whole column is NaN
                power_filt.loc[:, i] = 0    

            else:
                power_filt.loc[nan, i] = np.interp(np.flatnonzero(nan), np.flatnonzero(~nan), power_filt.loc[~nan, i])

        
        # integrate values (after interpolation) to get power of full frequency bands
        delta_power = simps(power_filt.iloc[:,deltaidx[0]:deltaidx[1]], dx=freq_res)
        delta_smoothed = gaussian_filter(delta_power, sigma=10, mode='wrap')

        alpha_power = simps(power_filt.iloc[:,alphaidx[0]:alphaidx[1]], dx=freq_res)
        alpha_smoothed = gaussian_filter(alpha_power, sigma=10, mode='wrap')

        if not np.any(alpha_power):
            return
        theta_power = simps(power_filt.iloc[:,thetaidx[0]:thetaidx[1]], dx=freq_res)
        total_power = simps(power_filt, dx=freq_res)

        # calculate alpha to theta+delta ratio
        ratioAT = alpha_power/ (theta_power)
        ratioAT_smoothed = gaussian_filter(ratioAT, sigma=4, mode = 'wrap')

        ### find Delta peaks from intial mnima to avoid delta noise at sleep intiation         
        # Define a window size (+/- 45 minutes)
        # Calculate the local minima before and after each peak
        
        local_minima = []
        window_size = 90
        
        
        if True:
            
            if nan_flag:
                nanind = np.nonzero(rejected_epochs)[0].tolist()
                noisevals = rejected_epochs*1
                noise_p = []

                # sum 1's (noisy epochs) in 5-min windows
                for i in range(0,len(noisevals),10):
                    noise_p.append((noisevals[i:i+10]==1).sum())
                
                strt = self.find_position(noise_p)

                if strt is None:
                    print('couldn\'t find clean segment at the beginning')


                # the same but backwards
                noise_p = []
                clean_seg = []
                for i in range(len(noisevals),0,-10):
                    noise_p.append((noisevals[i-10:i]==1).sum())
                
                endind = rejected_epochs.shape[0]-self.find_position(noise_p)

#                 endind = None
#                 if all(n < 8 for n in noise_p[:5]):
#                     endind = rejected_epochs.shape[0]
#                 else:
#                     for j in range(4, len(noise_p)):
#                         if noise_p[j] < 8:
#                             clean_seg.append(j)
#                         elif noise_p[j] >= 8:
#                             clean_seg = []


#                         if len(clean_seg) == 2:
#                             endind = rejected_epochs.shape[0]-clean_seg[0]*10
#                             break

                if endind is None:
                    print('couldn\'t find clean segment at the end')    

                print('Start: '+str(strt))
                print('End: '+str(endind))
                
            else:
                strt = 0
                endind = rejected_epochs.shape[0]
                
            ### Calculate delta peaks    
            # get median of delta power
            minAmp = np.median(delta_smoothed)
            # get 5th and 95th perentiles for prominence calculation
            perc = np.percentile(delta_smoothed, [5, 95])
            if bool(prom_flag):
                prom = (perc[1]-perc[0])/3
            else:
                prom = None
                
            peaks, _ = find_peaks(delta_smoothed[strt:endind], width = 20, height = minAmp, prominence = prom, distance = 60)

            if len(peaks)<2 and bool(prom_flag):
                prom = (perc[1]-perc[0])/5
                peaks5, _ = find_peaks(delta_smoothed[strt:endind], width = 20, height = minAmp, prominence = prom, distance = 60)
                try:
                    peaks = peaks5[peaks5>=peaks[0]]
                except:
                    peaks = peaks
                    
            
            ### Calculate alpha to delta+theta ratio peaks    
            ratio_peaks, _ = find_peaks(ratioAT_smoothed[strt:endind], width = 20, height = np.percentile(ratioAT_smoothed[strt:endind],95))
            peaks = peaks + strt
            ratio_peaks = ratio_peaks + strt
            
            # find local minima before and after each Delta peak within a 45 min window
            for peak in peaks:
                # Local minima before the peak minimum is 5 minutes after recording statrted
                before_min = max(strt+10, peak - window_size)
                local_minima.append(np.argmin(delta_smoothed[before_min:peak]) + before_min)

                # Local minima after the peak
                after_min = min(endind, peak + window_size)
                local_minima.append(np.argmin(delta_smoothed[peak:after_min]) + peak)
            
            print('peaks: '+ str(peaks))
            print('local minima '+ str(local_minima))
        
            # find local minima before and after each Ratio peak within a 30 min window
            window_size = 60
            local_minimaAT = []   
            for peak in ratio_peaks:
                before_min = max(strt, peak - window_size)
                local_minimaAT.append(np.argmin(ratioAT_smoothed[before_min:peak])+before_min)
                
                after_min = min(endind, peak + window_size)
                local_minimaAT.append(np.argmin(ratioAT_smoothed[peak:after_min]) + peak)
                
            print('local_minimaAT'+str(local_minimaAT))
        
        #define threshold calculation window as 10 minutes after local minima from first SWA cycle
        
        # Detect false identification of local minima 
                
       
        clean_ref_window = []
        i = local_minima[0]
        while len(clean_ref_window) < 20 and i < len(ratioAT):
            if not rejected_epochs[i]:
                clean_ref_window.append(i)
            i += 1
        
        ### find sleep_onet based on alpha/theta ration ---------
        # Initialize variables for sleep-onset
        so_algo = None
        consecutive_count_so = 0
        target_value = (np.mean(ratioAT_smoothed[clean_ref_window]) + np.std(ratioAT_smoothed[clean_ref_window]))
        
        # start from the first Delta local minima and iterate bachward to find 10 consecutive epochs 
        # with ratio > threshold. The final epochs within these epochs is sleep onset.
        i = local_minima[0]-1
        while consecutive_count_so < 10 and i >= max((local_minima[0]-40),strt):
            epoch = ratioAT_smoothed[i]
            if epoch >= target_value:
                consecutive_count_so +=1
                if consecutive_count_so == 10:
                    so_algo = i+9
                    so_flag = 'cons'
                    break
            else:
                consecutive_count_so = 0
            i -= 1
        
        # if 10 consecutive epochs weren't detected:
        if (so_algo==0) | (so_algo is None):
            print('failed loop so')            
            try:
                # define sleep onset as the last epoch prior to the first delta local minima with ratio > threshold
                so_algo = strt + np.where(ratioAT[strt:local_minima[0]] >= target_value)[0][-1]
                print('logic')
                so_flag = 'logic'
            except: 
                # if nothing worked, define sleep onset as the first delta local minima
                so_algo = local_minima[0]
                so_flag = 'local_minima'
                print('alternative for so didnt work, setting local minima')
               
                
        ### find Final awaknning based on alpha/theta ratio -----
        
        target_value = (np.mean(ratioAT_smoothed[clean_ref_window]) - np.std(ratioAT_smoothed[clean_ref_window]))

        consecutive_count = 0  # Variable to count consecutive epochs meeting the condition        
        end_wu = endind-1
        print(f'end_wu:{endind}')
        # Initialize wu_algo to None. Will update this if we find a match.
        wu_algo = None
        
        ratio_pre = []
        ratio_post = []
        i_index = []
        first_clean = None
        # turn boolean noise mark to numeric
        noisevals = rejected_epochs*1

        # # get position of clean epochs
        clean_indices = np.where(noisevals==0)[0]
        # get differences between indices 
        diff_indices = np.diff(clean_indices)
        
        
#         for i in range(len(diff_indices), 0, -1):
#             if (diff_indices[i-19:i]==1).all():
#                 first_clean = i-19
#                 break

        for i in range(len(noisevals), 0, -1):
            if noisevals[i-30:i].sum() <= 10 :
                first_clean = i-30
                print('new_logic')
                break
        
        if first_clean is not None:
            # loop from the last clean segment to end, and update mean ratio of 5 epochs before and after each epoch
            for i in range(first_clean, noisevals.shape[0]-5):
                ratio_pre.append(ratioAT_smoothed[i-5:i].mean())
                ratio_post.append(ratioAT_smoothed[i:i+5].mean())
                i_index.append(i)

            #calculate the difference between pre and post ratios
            ratio_dif = [a-b for a,b in zip(ratio_post,ratio_pre)]
            wu_algo = i_index[ratio_dif.index(max(ratio_dif))]
            wu_flag = 'new_logic'

        # if wu_algo is None:

#             # Loop through the array in reverse, starting from the end
#             for i in range(local_minimaAT[-2],  first_clean, -1):
#                 epoch =  ratioAT_smoothed[i]

#                 # Check if the epoch meets the condition
#                 if epoch <= target_value:
#                     if True:
#                         consecutive_count += 1  # Increment the count
#                         if consecutive_count == 10:
#                             wu_algo = i+9  # Found the start of the 10 consecutive epochs
#                             wu_flag = 'cons'
#                             print(wu_algo)
#                             break  # Exit the loop
#                 else:
#                     consecutive_count = 0  # Reset the count

#             # if no final awakening was found:
#         if wu_algo is None:
        
#             print('failed loop wu')
#             try: #define final awakening as the last epoch with ratio < threshold
#                 wu_algo = local_minimaAT[-2] + np.where(ratioAT[local_minimaAT[-2]:first_clean] <= target_value)[0][-1] #change to this! 30/10
#                 wu_flag = 'logic'
#                 print('logic')
#             except:
#                 try:
#                     wu_algo = local_minimaAT[-2] + np.argmax(ratioAT[local_minimaAT[-2]:end_wu])-1
#                     wu_flag = 'local_max'
#                     print('local_max')
#                 except: # if nothing was found, define final awakening as the last clean epoch
#                     print('alternative for wu didnt work, setting end_wu')
#                     wu_flag = 'end_wu1'
#                     print('end_wu1')
#                     wu_algo = end_wu
        
#         # If the final awakening that was found precedes the last Delta local minima, we redefine final awakening as the last
#         if wu_algo <= local_minima[-2]:
            
#             wu_algo = end_wu
#             wu_flag = wu_flag+'_end_wu2'

        print([so_algo, wu_algo])
                
                                                     
        
        # return alpha_smoothed, delta_smoothed, ratioAT_smoothed, target_value, wu_algo
        return so_algo, wu_algo, delta_smoothed, ratioAT, alpha_power, delta_power, theta_power, peaks, so_flag,  wu_flag, local_minima
        # return so_algo, wu_algo, delta_smoothed, ratioAT, alpha_power, delta_power, theta_power, peaks, so_flag,  wu_flag, local_minima, clean_ref_window, ratio_peaks, local_minimaAT, ratioAT_smoothed, total_power
        # return so_algo, wu_algo, delta_smoothed, ratioAT, alpha_power, delta_power, theta_power, peaks, local_minima, local_minimaAT, ratio_peaks, ratioAT_smoothed