%% FIND functional/structural files
% note: this will look for all data in these folders, irrespestive of the specific download subsets entered as command-line arguments
% matlab -nodisplay -nosplash -nodesktop -r "run('01a_V2V_analysis_parallel.m');exit;"

% in the following line replace the path with the one from `s01_preprocessing_data.ipynb`
directory= '/your/path/';
NSUBJECTS=14;

% if needed, add conn and spm12 to the path
% addpath("/path/to/conn")
% addpath("/path/to/spm12")
cwd=pwd;
FUNCTIONAL_FILE={};
STRUCTURAL_FILE={};
cd(directory)

tFUNCTIONAL_FILE=cellstr(conn_dir(fullfile('sub-0*bold.nii.gz')));
tSTRUCTURAL_FILE=cellstr(conn_dir(fullfile('sub-0*T1w.nii.gz')));
    FUNCTIONAL_FILE=[FUNCTIONAL_FILE;tFUNCTIONAL_FILE(:)];
    STRUCTURAL_FILE=[STRUCTURAL_FILE;tSTRUCTURAL_FILE(:)];

    if ~NSUBJECTS, NSUBJECTS=length(STRUCTURAL_FILE); end
if rem(length(FUNCTIONAL_FILE),NSUBJECTS),error('mismatch number of functional files %n', length(FUNCTIONAL_FILE));end
if rem(length(STRUCTURAL_FILE),NSUBJECTS),error('mismatch number of anatomical files %n', length(FUNCTIONAL_FILE));end
nsessions=length(FUNCTIONAL_FILE)/NSUBJECTS;
FUNCTIONAL_FILE=reshape(FUNCTIONAL_FILE,[NSUBJECTS,nsessions]);
STRUCTURAL_FILE={STRUCTURAL_FILE{1:NSUBJECTS}};
disp([num2str(size(FUNCTIONAL_FILE,1)),' subjects']);
disp([num2str(size(FUNCTIONAL_FILE,2)),' sessions']);
TR=1.4; % Repetition time = 2 seconds


%% Prepares batch structure
clear batch;
batch.filename=fullfile(cwd,'LEMON_conn_preprocessing.mat');            % New conn_*.mat experiment name
batch.parallel.N=NSUBJECTS;                             % One process per subject
batch.parallel.profile='Background';                         % Grid Engine profile (change to your cluster settings and uncomment this line to use a non-default profile)

%% SETUP & PREPROCESSING step (using default values for most parameters, see help conn_batch to define non-default values)
% CONN Setup                                            % Default options (uses all ROIs in conn/rois/ directory); see conn_batch for additional options 
% CONN Setup.preprocessing                               (realignment/coregistration/segmentation/normalization/smoothing)
batch.Setup.isnew=1;
batch.Setup.nsubjects=NSUBJECTS;
batch.Setup.RT=TR;                                        % TR (seconds)
batch.Setup.functionals=repmat({{}},[NSUBJECTS,1]);       % Point to functional volumes for each subject/session
for nsub=1:NSUBJECTS,for nses=1:nsessions,batch.Setup.functionals{nsub}{nses}=FUNCTIONAL_FILE{nsub,nses}; end; end %note: each subject's data is defined by three sessions and one single (4d) file per session
batch.Setup.structurals=STRUCTURAL_FILE;                  % Point to anatomical volumes for each subject
nconditions=nsessions;                                  % treats each session as a different condition (comment the following three lines and lines 84-86 below if you do not wish to analyze between-session differences)
if nconditions==1
    batch.Setup.conditions.names={'rest'};
    for ncond=1,for nsub=1:NSUBJECTS,for nses=1:nsessions,              batch.Setup.conditions.onsets{ncond}{nsub}{nses}=0; batch.Setup.conditions.durations{ncond}{nsub}{nses}=inf;end;end;end     % rest condition (all sessions)
else
    batch.Setup.conditions.names=[{'rest'}, arrayfun(@(n)sprintf('Session%d',n),1:nconditions,'uni',0)];
    for ncond=1,for nsub=1:NSUBJECTS,for nses=1:nsessions,              batch.Setup.conditions.onsets{ncond}{nsub}{nses}=0; batch.Setup.conditions.durations{ncond}{nsub}{nses}=inf;end;end;end     % rest condition (all sessions)
    for ncond=1:nconditions,for nsub=1:NSUBJECTS,for nses=1:nsessions,  batch.Setup.conditions.onsets{1+ncond}{nsub}{nses}=[];batch.Setup.conditions.durations{1+ncond}{nsub}{nses}=[]; end;end;end
    for ncond=1:nconditions,for nsub=1:NSUBJECTS,for nses=ncond,        batch.Setup.conditions.onsets{1+ncond}{nsub}{nses}=0; batch.Setup.conditions.durations{1+ncond}{nsub}{nses}=inf;end;end;end % session-specific conditions
end
batch.Setup.preprocessing.steps='default_mni';
batch.Setup.preprocessing.sliceorder='interleaved (Siemens)';
batch.Setup.done=1;
batch.Setup.overwrite='Yes';                            


%% DENOISING step
% CONN Denoising                                    % Default options (uses White Matter+CSF+realignment+conditions as confound regressors); see conn_batch for additional options 
batch.Denoising.filter=[0.009, 0.08];                 % frequency filter (band-pass values, in Hz)
batch.Denoising.done=1;
batch.Denoising.overwrite='Yes';


%% Run all analyses
conn_batch(batch);
