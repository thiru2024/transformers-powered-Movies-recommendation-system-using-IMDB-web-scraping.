# transformers-powered-Movies-recommendation-system-using-IMDB-web-scraping.

## Overview
This project implements a movie recommendation system using BERT embeddings to process and analyze user inputs. It allows users to specify their preferences through natural language queries, and the system recommends movies based on the provided input. The system also incorporates IMDb ratings and previously mentioned movie preferences to refine recommendations.

## Features
1. **User Input Parsing**:
   - Allows users to specify movie names within double inverted commas (e.g., "Inception").
   - IMDb ratings can be mentioned as `imdb:<rating>` (e.g., `imdb:8.2`).
2. **BERT Embeddings**:
   - Utilizes the BERT model to create embeddings for movie overviews and user input.
3. **Similarity Calculation**:
   - Computes weighted cosine similarity between user input and movie embeddings to recommend movies.
4. **Customizable Recommendations**:
   - Users can set a minimum IMDb rating and include specific movies to personalize recommendations.
5. **Top Movie Suggestions**:
   - Recommends the top 15 movies based on similarity scores.

## Requirements
- Python 3.7+
- Required Libraries:
  - pandas
  - numpy
  - torch
  - transformers
  - scikit-learn
  - re

## Dataset
The project uses a CSV file named `imdb_top_1000.csv` containing metadata about movies. The dataset should include at least the following columns:
- `Genre`
- `Series_Title`
- `IMDB_Rating`
- `Overview`

## Installation
1. Clone this repository or download it directly from github using this link. 'https://github.com/thiru2024/transformers-powered-Movies-recommendation-system-using-IMDB-web-scraping.'
2. Install the required libraries using pip:
   ```bash
   pip install pandas numpy torch transformers scikit-learn
   ```
3. Download the `imdb_top_1000.csv` dataset and place it in the appropriate directory. Update the `file_path` variable in the script to match the dataset's location.

## Usage
1. Run the script using:
   ```bash
   python trans2.py
   ```
2. Follow the prompts:
   - Mention movie names within double inverted commas (e.g., "Inception").
   - Specify IMDb rating with the format `imdb:<rating>` (e.g., `imdb:8.2`).
3. The system will display the top 15 recommended movies.

## Code Walkthrough

### Data Loading and Preprocessing
- The CSV file is loaded using pandas and filtered to retain relevant columns (`Genre`, `Series_Title`, `IMDB_Rating`, `Overview`).

### BERT Model Integration
- The `transformers` library is used to load the BERT tokenizer and model (`bert-base-uncased`).
- Movie overviews are tokenized and converted into BERT embeddings.

### User Input Processing
- The input is preprocessed to normalize the text and extract movie names and IMDb ratings.

### Similarity Calculation
- Cosine similarity is calculated between user input embeddings and movie embeddings.
- Weighted similarity scores are computed if the user specifies movies they like.

### Recommendation
- The top 15 movies with the highest similarity scores are returned as recommendations.

## Example Input and Output
### Input
- User prompt:
  ```
  "Inception" imdb:8
  ```

### Output
- Recommended Movies:
  ```
  1. The Dark Knight
  2. Interstellar
  3. Fight Club
  4. ...
  ```

## Customization
- **Adjusting IMDb Filter**: Modify the `extract_imdb_rating` function to change the behavior of IMDb filtering.
- **Number of Recommendations**: Update the number of top movies in the `recommend_movies` function.

## Limitations
- The system relies on the quality and coverage of the `imdb_top_1000.csv` dataset.
- BERT embeddings are computationally intensive for large datasets.

## Future Enhancements
- Incorporate additional metadata like genres and cast to improve recommendations.
- Optimize embeddings using a fine-tuned BERT model for movie data.
- Allow for more complex queries, including genre preferences or keyword-based filtering.

## Acknowledgments
- The dataset used in this project is sourced from [IMDb](https://www.imdb.com).
- The BERT model is provided by the [Hugging Face Transformers](https://huggingface.co/transformers/) library.

