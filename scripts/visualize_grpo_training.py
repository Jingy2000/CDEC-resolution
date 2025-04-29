#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize GRPO training results')
    parser.add_argument('--checkpoint_dir', type=str, default='models/grpo/checkpoint-1000',
                        help='Directory containing trainer_state.json')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save output figures')
    parser.add_argument('--window_size', type=int, default=10,
                        help='Window size for rolling averages')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures')
    parser.add_argument('--exclude_components', type=str, nargs='+', default=['int_reward_func'],
                        help='Reward components to exclude from visualization')
    parser.add_argument('--figsize_main', type=float, nargs=2, default=[24, 14],
                        help='Figure size for main metrics plot (width, height)')
    parser.add_argument('--figsize_reward', type=float, nargs=2, default=[14, 8],
                        help='Figure size for reward composition plot (width, height)')
    parser.add_argument('--figsize_progression', type=float, nargs=2, default=[14, 6],
                        help='Figure size for reward progression plot (width, height)')
    return parser.parse_args()

def load_training_data(checkpoint_dir):
    filepath = os.path.join(checkpoint_dir, 'trainer_state.json')
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def visualize_training(data, args):
    # Extract training history
    log_history = data['log_history']
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(log_history)
    
    # Set style for better visualization
    plt.style.use('ggplot')
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ===== Figure 1: Main training metrics =====
    fig = plt.figure(figsize=tuple(args.figsize_main))
    fig.suptitle('GRPO Training Metrics', fontsize=24)
    
    # 1. Plot reward over time
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df['step'], df['reward'], 'b-', linewidth=2)
    ax1.fill_between(df['step'], 
                    df['reward'] - df['reward_std'], 
                    df['reward'] + df['reward_std'], 
                    alpha=0.3, color='b')
    ax1.set_title('Overall Reward')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward')
    
    # 2. Plot different reward components - excluding specified components
    ax2 = plt.subplot(2, 2, 2)
    reward_components = [col for col in df.columns if col.startswith('rewards/') and 
                         not any(exclude in col for exclude in args.exclude_components)]
    
    for reward_type in reward_components:
        label = reward_type.replace('rewards/', '').replace('_reward_func', '')
        ax2.plot(df['step'], df[reward_type], label=label, linewidth=2)
    
    ax2.set_title('Reward Components')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Component Value')
    ax2.legend()
    
    # 3. Plot completion length
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(df['step'], df['completion_length'], 'g-', linewidth=2)
    ax3.set_title('Completion Length')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Length')
    
    # 4. Plot learning rate
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(df['step'], df['learning_rate'], 'c-', linewidth=2)
    ax4.set_title('Learning Rate')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Learning Rate')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # ===== Figure 2: Reward composition =====
    plt.figure(figsize=tuple(args.figsize_reward))
    
    # Create a stacked area chart for the absolute values of reward components
    plt.stackplot(df['step'], 
                  [df[comp] for comp in reward_components],
                  labels=[comp.replace('rewards/', '').replace('_reward_func', '') for comp in reward_components],
                  alpha=0.7)
    
    plt.plot(df['step'], df['reward'], 'k-', linewidth=2, label='Total Reward')
    plt.title('Composition of Total Reward Over Time', fontsize=18)
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Reward Value', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True)
    
    # ===== Figure 3: Reward progression =====
    plt.figure(figsize=tuple(args.figsize_progression))
    
    # Calculate rolling averages
    window_size = args.window_size
    rolling_reward = df['reward'].rolling(window=window_size).mean()
    
    plt.plot(df['step'], df['reward'], 'b-', alpha=0.3, label='Raw Reward')
    plt.plot(df['step'], rolling_reward, 'b-', linewidth=3, label=f'Rolling Avg (window={window_size})')
    plt.title('Reward Progression During Training', fontsize=18)
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True)
    
    # Save the figures
    output_files = []
    
    plt.figure(1)
    output_path = os.path.join(args.output_dir, 'grpo_training_metrics.png')
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    output_files.append(output_path)
    
    plt.figure(2)
    output_path = os.path.join(args.output_dir, 'grpo_reward_composition.png')
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    output_files.append(output_path)
    
    plt.figure(3)
    output_path = os.path.join(args.output_dir, 'grpo_reward_progression.png')
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    output_files.append(output_path)
    
    return output_files

def main():
    args = parse_args()
    data = load_training_data(args.checkpoint_dir)
    output_files = visualize_training(data, args)
    
    print("Visualization complete!")
    print(f"Generated {len(output_files)} figures:")
    for file in output_files:
        print(f"- {file}")

if __name__ == "__main__":
    main() 