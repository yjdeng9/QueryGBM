import sys
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from statannotations.Annotator import Annotator

import matplotlib.pyplot as plt
import seaborn as sns


def fold_model(X, y, n_splits=5,n_iters=100):
    accuracies = []
    roc_aucs = []
    for i in range(n_iters):
        acc_ = []
        auc_ = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            acc_.append(accuracy)
            auc_.append(roc_auc)
        accuracies.append(np.mean(acc_))
        roc_aucs.append(np.mean(auc_))
    print(f'Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}')
    print(f'Mean ROC AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}')
    return accuracies, roc_aucs



def main():

    embedding_path = sys.argv[1]
    clic_onehot_path = sys.argv[2]
    gen_pca_path = sys.argv[3]
    label_path = sys.argv[4]

    result_dir = "results_xgboost"
    os.makedirs(result_dir, exist_ok=True)

    label_df = pd.read_csv(label_path)
    label_y = label_df["OS"]

    all_embedding = np.load(embedding_path)
    clinc_onehot = pd.read_csv(clic_onehot_path).drop(columns=["patient_id"])
    gen_embedding = pd.read_csv(gen_pca_path).drop(columns=["patient_id"])
    all_embedding = pd.DataFrame(all_embedding, columns=[f'emb_{i}' for i in range(all_embedding.shape[1])])
    clinc_onehot.columns = ['onehot_%d'%i for i in range(clinc_onehot.shape[1])]

    print('Training with embedding:')
    accuracies, roc_aucs = fold_model(all_embedding, label_y, n_splits=5, n_iters=20)
    np.savez_compressed(os.path.join(result_dir, 'clinicalEmbedding_results.npz'), accuracies=accuracies, roc_aucs=roc_aucs)

    print('Training with clinical onehot only:')
    accuracies, roc_aucs = fold_model(clinc_onehot, label_y, n_splits=5, n_iters=20)
    np.savez_compressed(os.path.join(result_dir, 'clinicalOnehot_results.npz'), accuracies=accuracies, roc_aucs=roc_aucs)

    print('Training with gene PCA only:')
    accuracies, roc_aucs = fold_model(gen_embedding, label_y, n_splits=5, n_iters=20)
    np.savez_compressed(os.path.join(result_dir, 'genePCA_results.npz'), accuracies=accuracies, roc_aucs=roc_aucs)

    combiend = pd.concat([gen_embedding, clinc_onehot], axis=1)
    print('Training with gene PCA + clinical onehot:')
    accuracies, roc_aucs = fold_model(combiend, label_y, n_splits=5, n_iters=20)
    np.savez_compressed(os.path.join(result_dir, 'genePCA_clinicalOnehot_results.npz'), accuracies=accuracies, roc_aucs=roc_aucs)



def plot_result():
    accuracies_emb, roc_aucs_emb = np.load("results_xgboost/clinicalEmbedding_results.npz").values()
    accuracies_cli_onehot, roc_aucs_cli_onehot = np.load("results_xgboost/clinicalOnehot_results.npz").values()
    accuracies_gen, roc_aucs_gen = np.load("results_xgboost/genePCA_results.npz").values()
    accuracies_gen_cli_onehot, roc_aucs_gen_cli_onehot = np.load("results_xgboost/genePCA_clinicalOnehot_results.npz").values()

    data = {
        "Model": (
                ["Clinical Embedding"] * len(accuracies_emb) +
                ["Clinical Onehot"] * len(accuracies_cli_onehot) +
                ["Gene PCA"] * len(accuracies_gen) +
                ["Gene PCA + Clinical Onehot"] * len(accuracies_gen_cli_onehot)
        ),
        "Accuracy": (
                list(accuracies_emb) +
                list(accuracies_cli_onehot) +
                list(accuracies_gen) +
                list(accuracies_gen_cli_onehot)
        ),
        "ROC_AUC": (
                list(accuracies_emb) +
                list(roc_aucs_cli_onehot) +
                list(roc_aucs_gen) +
                list(roc_aucs_gen_cli_onehot)
        )
    }

    df = pd.DataFrame(data)

    # 画 Accuracy 的箱线图
    plt.figure(figsize=(6, 5))
    ax = sns.boxplot(x="Model", y="Accuracy", data=df, palette="Set2")
    sns.stripplot(x="Model", y="Accuracy", data=df, color="black", size=4, alpha=0.6)

    # 需要比较的组对
    pairs = [
        ("Clinical Embedding", "Clinical Onehot"),
        ("Clinical Embedding", "Gene PCA"),
        ("Gene PCA", "Gene PCA + Clinical Onehot")
    ]

    annotator = Annotator(ax, pairs, data=df, x="Model", y="Accuracy")
    annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
    annotator.apply_and_annotate()

    plt.xticks(rotation=20)
    plt.title("Accuracy Comparison across Models")
    plt.tight_layout()
    plt.show()

    # 画 ROC AUC 的箱线图
    plt.figure(figsize=(6, 5))
    ax = sns.boxplot(x="Model", y="ROC_AUC", data=df, palette="Set2")
    sns.stripplot(x="Model", y="ROC_AUC", data=df, color="black", size=4, alpha=0.6)

    pairs_auc = [
        ("Clinical Embedding", "Clinical Onehot"),
        ("Clinical Embedding", "Gene PCA"),
        ("Gene PCA", "Gene PCA + Clinical Onehot")
    ]

    annotator = Annotator(ax, pairs_auc, data=df, x="Model", y="ROC_AUC")
    annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
    annotator.apply_and_annotate()

    plt.xticks(rotation=20)
    plt.title("ROC AUC Comparison across Models")
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
    plot_result()
