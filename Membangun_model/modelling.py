import os
import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    # ==========================================
    # 1. INISIALISASI SAMBUNGAN DAGSHUB
    # ==========================================
    print("🌐 Menyambungkan ke remote tracker DagsHub...")
    dagshub.init(repo_owner="ariefwcks303", repo_name="Eksperimen_SML_Arief", mlflow=True)
    
    # Menamai eksperimen di server DagsHub
    mlflow.set_experiment("Student_Placement_Baseline")
    
    # Mengaktifkan Autolog (Fitur otomatis bawaan MLflow)
    mlflow.autolog()

    # ==========================================
    # 2. MEMBACA DATASET
    # ==========================================
    data_path = "student_dataset_clean.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"⚠️ Berkas {data_path} tidak ditemukan di folder ini!")
        
    df = pd.read_csv(data_path)
    
    # Memisahkan Fitur dan Target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Membagi data (Train 80%, Test 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ==========================================
    # 3. TRAINING MODEL & AUTOLOGGING
    # ==========================================
    print("⏳ Memulai proses training model baseline...")
    with mlflow.start_run(run_name="Random_Forest_Autolog"):
        # Membuat objek model Random Forest standar
        model = RandomForestClassifier(random_state=42)
        
        # Proses training (saat .fit() berjalan, autolog otomatis mencatat semuanya ke DagsHub)
        model.fit(X_train, y_train)
        
        print("✅ Model Baseline berhasil dilatih dan dicatat di DagsHub!")

if __name__ == "__main__":
    main()