import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, label_encoder, savefig=False, filename="cm", imp=False, rf_model=None, feature_names=[]):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if savefig:
        plt.savefig(f"./plots/confusion/{filename}.png", dpi=600)
        plt.close()
    else:
        plt.show()

    if imp:
        importances = rf_model.feature_importances_

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        top_features = importance_df.sort_values(by='Importance', ascending=False).head(20)

        plt.figure(figsize=(12, 8))
        plt.barh(top_features['Feature'], top_features['Importance'], color="skyblue")
        plt.gca().invert_yaxis()
        plt.xlabel("Feature Importance")
        plt.title("Top 20 Features by Importance")
        plt.tight_layout()
        if savefig:
            plt.savefig(f"./plots/feature_importance/{filename}.png", dpi=600)
            plt.close()
        else:
            plt.show()


def plot_losses(history, savefig=False, filename=""):

    val_losses = history.history["val_loss"]
    train_losses = history.history["loss"]

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    if savefig:
        plt.savefig(f"./plots/loss_acc/{filename}_loss.png", dpi=600)
        plt.close()
    else:
        plt.show()

    if "accuracy" in history.history:
        train_accuracies = history.history["accuracy"]
        val_accuracies = history.history["val_accuracy"]

        plt.figure(figsize=(10, 5))
        plt.plot(train_accuracies, label="Training Accuracy")
        plt.plot(val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        if savefig:
            plt.savefig(f"./plots/loss_acc/{filename}_accuracy.png", dpi=600)
            plt.close()
        else:
            plt.show()

def plot_results_table(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    table.auto_set_font_size(False)  
    table.set_fontsize(10)
    table.scale(1.5, 1.5)

    plt.show()