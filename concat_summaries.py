import os
import glob


def concatenate_summaries_clean(directory):
    """
    Concatenates all summarized text files into a single clean text file
    without headers or separators, suitable for LSTM training.

    Args:
        directory (str): Base directory path that contains 'summarized_texts_v3'

    Returns:
        str: Path to the concatenated file
    """
    # Construct the path to summarized_texts_v2 directory
    summary_dir = os.path.join(directory, 'summarized_texts_v3')

    # Check if directory exists
    if not os.path.exists(summary_dir):
        raise ValueError(f"Directory not found: {summary_dir}")

    # Get all summarized text files
    summary_files = glob.glob(os.path.join(summary_dir, "*_summarized.txt"))

    if not summary_files:
        raise ValueError(f"No summarized text files found in {summary_dir}")

    print(f"\nFound {len(summary_files)} summary files to concatenate")

    # Create output file
    output_path = os.path.join(summary_dir, 'combined_summaries_v3.txt')

    # Concatenate files
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for file_path in sorted(summary_files):
            with open(file_path, 'r', encoding='utf-8') as infile:
                content = infile.read().strip()
                outfile.write(content)
                # Add a single newline between documents
                outfile.write('\n')

    print(f"Combined text saved to: {output_path}")
    return output_path


# Usage example:
if __name__ == "__main__":
    try:
        directory = "/Users/yizu/Desktop/CS5100/FinalProject/mtps/data"
        output_path = concatenate_summaries_clean(directory)
        print("Successfully combined all summaries!")
    except Exception as e:
        print(f"Error: {str(e)}")
