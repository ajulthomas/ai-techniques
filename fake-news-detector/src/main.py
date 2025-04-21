import utils
import processing
import plots
import classic_models
import roberta_classifier


def main():
    df = utils.load_raw_data()
    df_clean = processing.preprocess_data(df)
    plots.plot_word_frequencies(df_clean, top_n=20)
    # classic_models.train_classic_models(df_clean)
    roberta_classifier.roberta_pretrained(df_clean)

if __name__ == "__main__":
    main()