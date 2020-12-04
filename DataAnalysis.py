# open all the Data

desired_attributes = ['title', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'liveness', 'tempo']
files = {}

directory = "/Users/eringrant/github/SpotifySupervisedAnalysis/Data"
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        files[os.path.join(directory, filename)] = 0
    else:
        continue

for name in files:
    file = open(name, "r")
    files[name] = pd.read_csv(file, delimiter=',', index_col=0)

# create summary statistics

# all the desired_attributes

#
