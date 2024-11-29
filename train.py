import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from lstm_crf import NERModel
import logging
from datetime import datetime
import os

def setup_logging(log_dir='logs'):
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def encode_sentences(sentences, word2idx, max_len):
    encoded = []
    for sentence in sentences:
        encoded_sentence = [word2idx.get(word.lower(), word2idx["<UNK>"]) for word in sentence]
        if len(encoded_sentence) < max_len:
            encoded_sentence += [word2idx["<PAD>"]] * (max_len - len(encoded_sentence))
        else:
            encoded_sentence = encoded_sentence[:max_len]
        encoded.append(encoded_sentence)
    return torch.tensor(encoded, dtype=torch.long)

def encode_labels(labels, tag2idx, max_len):
    encoded = []
    for label_seq in labels:
        encoded_label = [tag2idx.get(tag, tag2idx["O"]) for tag in label_seq]
        if len(encoded_label) < max_len:
            encoded_label += [tag2idx["<PAD>"]] * (max_len - len(encoded_label))
        else:
            encoded_label = encoded_label[:max_len]
        encoded.append(encoded_label)
    return torch.tensor(encoded, dtype=torch.long)

def main():
    # Set up logging
    logger = setup_logging()
    logger.info("Starting NER model training")

    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Parameters
    MAX_LEN = 50
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # Log parameters
    logger.info("Training Parameters:")
    logger.info(f"MAX_LEN: {MAX_LEN}")
    logger.info(f"EMBEDDING_DIM: {EMBEDDING_DIM}")
    logger.info(f"HIDDEN_DIM: {HIDDEN_DIM}")
    logger.info(f"BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"EPOCHS: {EPOCHS}")
    logger.info(f"LEARNING_RATE: {LEARNING_RATE}")

    # Load dataset
    logger.info("Loading CoNLL-2003 dataset...")
    try:
        dataset = load_dataset('conll2003', trust_remote_code=True)
        logger.info("Dataset loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

    # Create vocabularies
    logger.info("Creating vocabularies...")
    words = set()
    tags = set()
    for split in ['train', 'validation']:
        for sentence in dataset[split]:
            for word in sentence['tokens']:
                words.add(word.lower())
            for tag in sentence['ner_tags']:
                tags.add(tag)

    # Create word2idx and tag2idx
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word in sorted(words):
        word2idx[word] = len(word2idx)

    tag_names = dataset['train'].features['ner_tags'].feature.names
    tag2idx = {"<PAD>": 0}
    for tag in tag_names:
        tag2idx[tag] = len(tag2idx)
    idx2tag = {v: k for k, v in tag2idx.items()}

    logger.info(f"Vocabulary size: {len(word2idx)}")
    logger.info(f"Number of tags: {len(tag2idx)}")

    # Prepare data
    logger.info("Preparing training and validation data...")
    train_sentences = [example['tokens'] for example in dataset['train']]
    train_labels = [[tag_names[tag] for tag in example['ner_tags']] 
                   for example in dataset['train']]
    
    val_sentences = [example['tokens'] for example in dataset['validation']]
    val_labels = [[tag_names[tag] for tag in example['ner_tags']] 
                 for example in dataset['validation']]

    # Encode data
    logger.info("Encoding data...")
    X_train = encode_sentences(train_sentences, word2idx, MAX_LEN)
    y_train = encode_labels(train_labels, tag2idx, MAX_LEN)
    X_val = encode_sentences(val_sentences, word2idx, MAX_LEN)
    y_val = encode_labels(val_labels, tag2idx, MAX_LEN)

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")

    # Create model configuration
    config = {
        'vocab_size': len(word2idx),
        'tagset_size': len(tag2idx),
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'padding_idx': word2idx['<PAD>'],
        'learning_rate': LEARNING_RATE,
        'word2idx': word2idx,  # Save vocabularies in config
        'tag2idx': tag2idx
    }

    # Initialize model
    logger.info("Initializing model...")
    model = NERModel(config)

    # Create datasets and dataloaders
    logger.info("Creating data loaders...")
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=model.data_collator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=model.data_collator
    )

    # Train the model
    logger.info("Starting training...")
    try:
        model.train(train_loader, val_loader, EPOCHS, idx2tag)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

    # Save the model
    logger.info("Saving model...")
    try:
        model.save('ner_model.pt')
        # Test loading
        loaded_model = NERModel.load('ner_model.pt')
        logger.info("Model saved and loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to save/load model: {str(e)}")
        raise

if __name__ == "__main__":
    main()