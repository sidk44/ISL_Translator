import pandas as pd

# adjust this path if needed:
csv_path = '/Users/siddharthkms/ViTA-for-Indian-Sign-Language/model/keypoint_classifier/keypoint.csv'
df = pd.read_csv(csv_path)
counts = df['label'].value_counts()

print(counts)
