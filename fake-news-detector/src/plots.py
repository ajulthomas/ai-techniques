import matplotlib.pyplot as plt
from collections import Counter


# Count words separately for each class
def get_word_frequencies(df, class_label):
    class_tokens = df[df["label"] == class_label]["tokens"].explode()
    return Counter(class_tokens)


# define the function to plot word frequencies
def plot_word_frequencies(df, top_n=15):
    # Count words separately for each class
    fake_freq = get_word_frequencies(df, "FAKE")  # change to 0 if numeric
    real_freq = get_word_frequencies(df, "TRUE")  # change to 1 if numeric

    # Get most common words
    fake_common = fake_freq.most_common(top_n)
    real_common = real_freq.most_common(top_n)

    # Split into words and frequencies
    fake_words, fake_counts = zip(*fake_common)
    real_words, real_counts = zip(*real_common)

    # Plot side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    # Fake news plot
    axs[0].barh(fake_words[::-1], fake_counts[::-1], color="red")
    axs[0].set_title("Top Words in FAKE News")
    axs[0].set_xlabel("Frequency")
    # Real news plot
    axs[1].barh(real_words[::-1], real_counts[::-1], color="green")
    axs[1].set_title("Top Words in REAL News")
    axs[1].set_xlabel("Frequency")
    plt.tight_layout()
    plt.show()
