import tensorflow as tf
import numpy as np
import h5py
import os
from generate_datasets import irregular_gen, regular_gen, open_gen, wider_line_gen, scrambled_gen, random_color_gen, filled_gen, lines_gen, arrows_gen
from tqdm import tqdm
import matplotlib.pyplot as plt

class MetaDatasetGenerator:
    def __init__(self):
        self.tqdm = tqdm
        self.support_sizes = [4, 6, 8, 10]  # Variable support sizes
        self.query_size = 3  # Fixed query size
        
        def create_batch_generator(gen_fn):
            def batch_gen(label, batch_size=32):
                images = []
                for _ in range(batch_size):
                    single_batch = next(gen_fn(batch_size=1, 
                                             category_type='same' if label else 'different'))
                    images.append(single_batch[0][0])
                return np.array(images)
            return batch_gen

        # Initialize generators with their respective types
        self.generators = {
            'original': self._load_original_svrt,
            'regular': create_batch_generator(regular_gen),
            'irregular': create_batch_generator(irregular_gen),
            'open': create_batch_generator(open_gen),
            'wider_line': create_batch_generator(wider_line_gen),
            'scrambled': create_batch_generator(scrambled_gen),
            'random_color': create_batch_generator(random_color_gen),
            'filled': create_batch_generator(filled_gen),
            'lines': create_batch_generator(lines_gen),
            'arrows': create_batch_generator(arrows_gen)
        }
        
        # Loading and caching original SVRT data from tfrecords
        self._load_original_data()
    
    def _load_original_data(self):
        """Load and cache original SVRT dataset from train/val/test splits"""
        self.feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        
        # Load original SVRT data for each split
        self.original_data = {
            'train': {0: [], 1: []},  # Different: 0, Same: 1
            'val': {0: [], 1: []},
            'test': {0: [], 1: []}
        }
        
        # Load data for each split
        for split in ['train', 'val', 'test']:
            dataset = tf.data.TFRecordDataset(f'../data/original_{split}.tfrecords')
            for raw_record in dataset:
                example = tf.io.parse_single_example(raw_record, self.feature_description)
                image = tf.io.decode_png(example['image_raw'], channels=3)
                label = example['label']
                self.original_data[split][label.numpy()].append(image)
    
    def _load_original_svrt(self, label, split='train'):
        """Get a random image from original SVRT dataset with specified label and split"""
        images = self.original_data[split][label]
        return images[np.random.randint(len(images))]
    
    def _create_episode(self, generator_fn, num_support=4, num_query=3, split='train'):
        """Creates a single episode with support and query sets"""
        # Generate balanced pairs for support set
        support_pairs = []
        # Generate same pairs (num_support // 2)
        for _ in range(num_support // 2):
            if generator_fn != self._load_original_svrt:
                image_batch = generator_fn(1, batch_size=1)  
                image = np.clip(image_batch[0] * 255.0, 0, 255).astype(np.uint8)
            else:
                image = generator_fn(1, split)
            support_pairs.append((image, 1.0))
        
        # Generate different pairs (num_support // 2)
        for _ in range(num_support // 2):
            if generator_fn != self._load_original_svrt:
                image_batch = generator_fn(0, batch_size=1)
                image = np.clip(image_batch[0] * 255.0, 0, 255).astype(np.uint8)
            else:
                image = generator_fn(0, split)
            support_pairs.append((image, 0.0))
            
        # Generate balanced pairs for query set
        query_pairs = []
        # Generate same pairs (num_query // 2)
        for _ in range(num_query // 2):
            if generator_fn != self._load_original_svrt:
                image_batch = generator_fn(1, batch_size=1)
                image = np.clip(image_batch[0] * 255.0, 0, 255).astype(np.uint8)
            else:
                image = generator_fn(1, split)
            query_pairs.append((image, 1.0))
            
        # Generate different pairs (num_query // 2)
        for _ in range(num_query // 2):
            if generator_fn != self._load_original_svrt:
                image_batch = generator_fn(0, batch_size=1)
                image = np.clip(image_batch[0] * 255.0, 0, 255).astype(np.uint8)
            else:
                image = generator_fn(0, split)
            query_pairs.append((image, 0.0))
            
        # Shuffle the pairs while maintaining correspondence
        np.random.shuffle(support_pairs)
        np.random.shuffle(query_pairs)
        
        # Unzip the pairs into separate lists
        support_images, support_labels = zip(*support_pairs)
        query_images, query_labels = zip(*query_pairs)
        
        # Stack into arrays
        support_images = np.stack(support_images)
        query_images = np.stack(query_images)
        support_labels = np.array(support_labels, dtype=np.float32)
        query_labels = np.array(query_labels, dtype=np.float32)
            
        return {
            'support_images': support_images,
            'support_labels': support_labels,
            'query_images': query_images,
            'query_labels': query_labels
        }
    
    def generate_dataset(self, output_dir, episodes_per_split={'train': 1000, 'val': 200, 'test': 200}):
        """Generates meta-learning episodes with variable support sizes"""
        os.makedirs(output_dir, exist_ok=True)
        
        for dataset_name, generator_fn in self.generators.items():
            print(f"\nGenerating episodes for {dataset_name}...")
            
            for support_size in self.support_sizes:
                print(f"\nGenerating episodes with support size {support_size}...")
                
                for split in ['train', 'val', 'test']:
                    output_file = os.path.join(output_dir, 
                                             f'{dataset_name}_support{support_size}_{split}.h5')
                    
                    # Generate episodes
                    episodes = []
                    for _ in self.tqdm(range(episodes_per_split[split]), 
                                     desc=f"{split.capitalize()} episodes"):
                        episode = self._create_episode(
                            generator_fn, 
                            num_support=support_size,
                            num_query=self.query_size,
                            split=split
                        )
                        episodes.append(episode)
                    
                    # Save episodes
                    with h5py.File(output_file, 'w') as f:
                        f.create_dataset('support_images', 
                                       data=np.stack([ep['support_images'] for ep in episodes]))
                        f.create_dataset('support_labels', 
                                       data=np.stack([ep['support_labels'] for ep in episodes]))
                        f.create_dataset('query_images', 
                                       data=np.stack([ep['query_images'] for ep in episodes]))
                        f.create_dataset('query_labels', 
                                       data=np.stack([ep['query_labels'] for ep in episodes]))
                    
                    # Verify the saved data
                    self._verify_dataset(output_file, 
                                       f"{dataset_name}_support{support_size}", 
                                       split)
    
    def _verify_dataset(self, file_path, dataset_name, split_type, num_episodes_to_check=5):
        """Verify that the generated episodes are saved and loaded correctly"""
        print(f"\nVerifying {split_type} episodes for {dataset_name}...")
        
        with h5py.File(file_path, 'r') as f:
            # Get total number of episodes
            total_episodes = f['support_images'].shape[0]
            print(f"Total episodes: {total_episodes}")
            
            # Check random episodes
            for i in range(num_episodes_to_check):
                idx = np.random.randint(total_episodes)
                print(f"\nChecking episode {idx}:")
                
                # Load episode data
                support_images = f['support_images'][idx]
                support_labels = f['support_labels'][idx]
                query_images = f['query_images'][idx]
                query_labels = f['query_labels'][idx]
                
                # Verify shapes
                print(f"Support images shape: {support_images.shape}")  # Should be (16, 128, 128, 3)
                print(f"Support labels shape: {support_labels.shape}")  # Should be (16,)
                print(f"Query images shape: {query_images.shape}")      # Should be (8, 128, 128, 3)
                print(f"Query labels shape: {query_labels.shape}")      # Should be (8,)
                
                # Verify value ranges
                print(f"Image value range: [{support_images.min()}, {support_images.max()}]")
                print(f"Support labels distribution - Same: {np.sum(support_labels)}, Different: {len(support_labels) - np.sum(support_labels)}")
                print(f"Query labels distribution - Same: {np.sum(query_labels)}, Different: {len(query_labels) - np.sum(query_labels)}")
                
                # Visualize one support and one query image
                if i == 0:  # Only for the first episode we check
                    plt.figure(figsize=(10, 5))
                    
                    plt.subplot(1, 2, 1)
                    plt.imshow(support_images[0])
                    plt.title(f'Support Image (Label: {support_labels[0]})')
                    plt.axis('off')
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(query_images[0])
                    plt.title(f'Query Image (Label: {query_labels[0]})')
                    plt.axis('off')
                    
                    plt.savefig(os.path.join(os.path.dirname(file_path), f'{dataset_name}_{split_type}_sample.png'))
                    plt.close()

def verify_h5_dataset(data_dir='../data/meta_h5', num_episodes_to_check=5):
    """Standalone verification function for HDF5 datasets"""
    print("Verifying HDF5 datasets...")
    
    dataset_types = ['original', 'regular', 'irregular', 'open', 'wider_line', 
                    'scrambled', 'random_color', 'filled', 'lines', 'arrows']
    
    for dataset_type in dataset_types:
        print(f"\nChecking {dataset_type} dataset:")
        
        # Check both train and test files
        for split in ['train', 'test']:
            file_path = os.path.join(data_dir, f'{dataset_type}_{split}.h5')
            
            try:
                with h5py.File(file_path, 'r') as f:
                    print(f"\n{split.capitalize()} set:")
                    print(f"Total episodes: {f['support_images'].shape[0]}")
                    
                    # Check random episodes
                    for i in range(num_episodes_to_check):
                        idx = np.random.randint(f['support_images'].shape[0])
                        
                        # Load episode data
                        support_images = f['support_images'][idx]
                        support_labels = f['support_labels'][idx]
                        query_images = f['query_images'][idx]
                        query_labels = f['query_labels'][idx]
                        
                        print(f"\nEpisode {idx}:")
                        print(f"Support images shape: {support_images.shape}")
                        print(f"Support labels shape: {support_labels.shape}")
                        print(f"Query images shape: {query_images.shape}")
                        print(f"Query labels shape: {query_labels.shape}")
                        print(f"Image value range: [{support_images.min()}, {support_images.max()}]")
                        print(f"Label distribution - Same: {np.sum(support_labels)}, Different: {len(support_labels) - np.sum(support_labels)}")
                        
            except Exception as e:
                print(f"Error loading {split} dataset for {dataset_type}: {str(e)}")

if __name__ == '__main__':
    # Create output directory for variable support size data
    output_dir = 'data/meta_h5'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting variable support size dataset generation...")
    
    episodes_per_split = {
        'train': 1000,
        'val': 200,
        'test': 200
    }
    
    # Create generator and generate datasets
    generator = MetaDatasetGenerator()
    generator.generate_dataset(output_dir, episodes_per_split=episodes_per_split)
    
    # Verify the generated episodes
    print("\nVerifying generated episodes...")
    verify_h5_dataset(data_dir='data/meta_h5')