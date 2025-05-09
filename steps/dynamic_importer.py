import pandas as pd
from zenml import step
import logging
import random
import string

# Configure logging
logger = logging.getLogger(__name__)


def generate_random_text(length: int) -> str:
    """Generates a random string of words."""
    words = [''.join(random.choices(string.ascii_lowercase,
                     k=random.randint(3, 10))) for _ in range(length)]
    return " ".join(words).capitalize() + "."


@step
def dynamic_importer(num_samples: int = 100, text_word_count: int = 150) -> pd.DataFrame:
    """
    Generates synthetic batch data for inference.
    The data will have 'text' and 'class' columns.
    'class' will be 0 (fake) or 1 (real).
    """
    logger.info(
        f"Starting synthetic data generation for {num_samples} samples...")

    data = []

    # More diverse vocabulary for text generation
    common_words = [
        "news", "report", "event", "update", "story", "article", "source", "information",
        "government", "politics", "election", "policy", "world", "local", "breaking",
        "economy", "business", "market", "finance", "company", "stock", "trade",
        "technology", "science", "research", "discovery", "innovation", "future",
        "health", "medicine", "virus", "pandemic", "vaccine", "care", "doctor",
        "sports", "game", "team", "player", "match", "score", "champion",
        "entertainment", "movie", "music", "celebrity", "show", "art", "culture",
        "people", "community", "life", "style", "food", "travel", "education",
        "climate", "environment", "nature", "energy", "weather", "disaster",
        "social", "media", "internet", "online", "platform", "user", "data",
        "analysis", "study", "expert", "opinion", "view", "comment", "discussion",
        "important", "significant", "key", "major", "new", "latest", "recent",
        "true", "false", "fact", "fiction", "claim", "evidence", "proof", "verify"
    ]

    for i in range(num_samples):
        # Generate random text
        current_text_word_count = random.randint(
            text_word_count - 50, text_word_count + 50)
        if current_text_word_count < 10:  # Ensure a minimum length
            current_text_word_count = 10

        text_words = random.choices(common_words, k=current_text_word_count)
        text = " ".join(text_words).capitalize() + "."

        # Assign a class label (0 for fake, 1 for real)
        news_class = random.choice([0, 1])

        data.append({
            'text': text,
            'class': news_class
        })

    batch_df = pd.DataFrame(data)
    logger.info(
        f"Successfully generated {len(batch_df)} synthetic data records with 'text' and 'class' columns.")
    logger.debug(
        f"Columns in the generated dataframe: {batch_df.columns.tolist()}")
    logger.debug(
        f"First 5 rows of the generated dataframe:\n{batch_df.head()}")

    return batch_df


if __name__ == "__main__":
    # This is for local testing of the step, if needed.
    logging.basicConfig(level=logging.INFO)
    generated_data = dynamic_importer(num_samples=5, text_word_count=20)
    if not generated_data.empty:
        print("\n--- Sample of generated data ---")
        print(generated_data)
        print(f"\nShape of generated data: {generated_data.shape}")
        print(
            f"\nClass distribution:\n{generated_data['class'].value_counts()}")
