import os
import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def main():
    # ==========================================
    # 1. INISIALISASI SAMBUNGAN DAGSHUB
    # ==========================================
    print("🌐 Menyambungkan ke remote tracker DagsHub (Mode Tuning)...")
    dagshub.init(repo_owner="ariefwcks303", repo_name="Eksperimen_SML_Arief", mlflow=True)
    
    # Menamai eksperimen Advance di server DagsHub
    mlflow.set_experiment("Student_Placement_Advance")
    
    # Read Dataset dari dalam folder yang sama
    df = pd.read_csv("student_dataset_clean.csv")
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # ==========================================
    # 2. HYPERPARAMETER TUNING (GRID SEARCH)
    # ==========================================
    print("⏳ Menjalankan Hyperparameter Tuning (GridSearch)...")
    rf = RandomForestClassifier(random_state=42)
    
    # Kombinasi parameter yang akan diuji
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
    # 3. MANUAL LOGGING (KRITERIA ADVANCED)
    # ==========================================
    with mlflow.start_run(run_name="Random_Forest_Manual_Tuning"):
        print("📝 Melakukan manual logging parameters & metrics ke DagsHub...")
        
        # A. Log Parameter Terbaik secara manual
        mlflow.log_params(grid_search.best_params_)
        
        # B. Log Metrik Evaluasi secara manual
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # C. Artefak 1: Membuat & Menyimpan Plot Confusion Matrix
        print("🖼️ Membuat visualisasi Confusion Matrix...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Placed', 'Placed'], yticklabels=['Not Placed', 'Placed'])
        plt.title('Confusion Matrix Final')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plot_path = "confusion_matrix.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Unggah plot ke DagsHub via MLflow
        mlflow.log_artifact(plot_path, artifact_path="plots")
        
        # D. Artefak 2: Menyimpan File Model Biner (.pkl)
        print("💾 Menyimpan model biner lokal...")
        model_file = "best_model.pkl"
        joblib.dump(best_model, model_file)
        
        # Unggah model ke DagsHub via MLflow
        mlflow.log_artifact(model_file, artifact_path="models")
        
        print("🚀 Berhasil! Semua data dan 2 jenis artefak telah terunggah ke DagsHub.")

if __name__ == "__main__":
    main()