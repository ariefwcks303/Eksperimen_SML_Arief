import os
import pandas as pd
import mlflow
import dagshub
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def main():
    # ==========================================
    # 1. INISIALISASI SAMBUNGAN DAGSHUB
    # ==========================================
    print("🌐 Menyambungkan ke remote tracker DagsHub (Mode Advanced)...")
    dagshub.init(repo_owner="ariefwcks303", repo_name="Eksperimen_SML_Arief", mlflow=True)
    
    # Menamai eksperimen Advance di server DagsHub
    mlflow.set_experiment("Student_Placement_Advance")
    
    # Membaca dataset yang berada di folder yang sama
    df = pd.read_csv("student_dataset_clean.csv")
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # ==========================================
    # 2. HYPERPARAMETER TUNING (GRID SEARCH)
    # ==========================================
    print("⏳ Menjalankan Hyperparameter Tuning (GridSearch)...")
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Hitung Metrik Evaluasi
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"📊 Hasil Akhir -> Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    
    # ==========================================
    # 3. MANUAL LOGGING (KRITERIA ADVANCED DICODING)
    # ==========================================
    with mlflow.start_run(run_name="Random_Forest_Manual_Tuning"):
        print("📝 Melakukan logging parameters, metrics & artifacts ke DagsHub...")
        
        # A. Log Parameter Terbaik hasil GridSearch
        mlflow.log_params(grid_search.best_params_)
        
        # B. Log Metrik Evaluasi Utama
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # C. Artefak 1: Simpan Model Biner (.pkl) secara Lokal lalu Unggah
        model_file = "best_model.pkl"
        joblib.dump(best_model, model_file)
        mlflow.log_artifact(model_file)
        
        # D. Artefak 2: Simpan Plot Visualisasi Confusion Matrix lalu Unggah
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Placed', 'Placed'], yticklabels=['Not Placed', 'Placed'])
        plt.title('Confusion Matrix Advanced Model')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plot_file = "confusion_matrix.png"
        plt.savefig(plot_file)
        plt.close()
        
        mlflow.log_artifact(plot_file)
        
        print("🚀 Berhasil! Semua data dan 2 jenis artefak telah terunggah ke DagsHub.")

if __name__ == "__main__":
    main()