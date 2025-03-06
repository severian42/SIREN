#!/usr/bin/env python3
"""
SIREN: Signal-Intelligent Resonance Encoding Network
A proof-of-concept implementation of IRE field equations for LLM memory enhancement.

This script creates a nonlocal field overlay that maintains information coherence
across a conversation, enabling "infinite context" without requiring the model to
process every token at each step.
"""

import numpy as np
import requests
import json
import time
import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional, Union
from scipy.ndimage import gaussian_filter, convolve
from scipy.spatial.distance import cosine, pdist, squareform
import hashlib
import csv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import torch
from tqdm import tqdm

# Create output directories if they don't exist
os.makedirs('SIREN/output', exist_ok=True)
os.makedirs('SIREN/output/metrics', exist_ok=True)
os.makedirs('SIREN/output/visualizations', exist_ok=True)
os.makedirs('SIREN/output/csv', exist_ok=True)
os.makedirs('SIREN/output/benchmarks', exist_ok=True)

class IRE_Field:
    """Implementation of the Information Relative Evolution field for memory representation"""
    
    def __init__(self, field_dims: Union[int, Tuple[int, ...]] = (128, 128), embedding_dim: int = 768,
                 diffusion_constant: float = 0.1, damping: float = 0.8, 
                 potential_alpha: float = 0.5, potential_beta: float = 0.1, 
                 nonlocal_scale: float = 100, projection_method: str = 'pca'):
        """
        Initialize the IRE field for memory representation
        
        Args:
            field_dims: Dimensions of the field representation (int for 1D field, tuple for multi-dimensional)
            embedding_dim: Dimension of the input embeddings
            diffusion_constant: Rate of information diffusion (D₀)
            damping: Damping coefficient (γ)
            potential_alpha: Strength of the organizing potential (α)
            potential_beta: Nonlinearity of the organizing potential (β)
            nonlocal_scale: Scale of nonlocal interactions
            projection_method: Method to project embeddings to field space ('pca', 'umap', 'direct')
        """
        # Parse field dimensions
        if isinstance(field_dims, int):
            self.field_dims = (field_dims,)  # Convert to 1D tuple
        else:
            self.field_dims = field_dims
            
        self.field_ndim = len(self.field_dims)  # Number of dimensions
        self.field_size = np.prod(self.field_dims)  # Total size
        self.embedding_dim = embedding_dim
        self.D = diffusion_constant
        self.gamma = damping
        self.alpha = potential_alpha
        self.beta = potential_beta
        self.nonlocal_scale = nonlocal_scale
        self.projection_method = projection_method
        
        # Initialize the field and velocity components
        self.psi = np.zeros(self.field_dims)  # Field values
        self.psi_t = np.zeros(self.field_dims)  # First time derivative
        
        # Create nonlocal kernel for each dimension
        self.kernels = self._create_nonlocal_kernels()
        
        # Time step for evolution (adjusted for stability)
        self.dt = 0.1  # Will be adjusted based on stability criteria
        
        # Semantic coordinates for each memory item
        self.semantic_coordinates = {}
        self.next_coordinate_id = 0
        
        # Projector for embedding -> field mapping
        self.projector = None
        self.has_enough_data_for_projection = False
        self.stored_embeddings = []
        
        # Metrics tracking
        self.energy_history = []
        self.amplitude_history = []
        self.evolution_steps = 0
    
    def _create_nonlocal_kernels(self) -> List[np.ndarray]:
        """Create nonlocal interaction kernels with excitation at short range and inhibition at longer range"""
        kernels = []
        
        for dim_size in self.field_dims:
            # Size of kernel should be proportional to the dimension
            kernel_size = min(dim_size // 4, 32)  # Limit kernel size for efficiency
            x = np.arange(-kernel_size, kernel_size+1)
            
            # Difference of Gaussians: short-range excitation, longer-range inhibition
            # Parameters adjusted for optimal pattern formation
            sigma_excite = max(2, kernel_size / 10)  # Short range excitation
            sigma_inhibit = max(10, kernel_size / 2)  # Longer range inhibition
            
            kernel = 2.0 * np.exp(-x**2/(2*sigma_excite**2)) - 0.5 * np.exp(-x**2/(2*sigma_inhibit**2))
            
            # Normalize kernel to ensure stability
            kernel = kernel / np.sum(np.abs(kernel))
            
            kernels.append(kernel)
        
        return kernels
    
    def _convolve_with_kernels(self, psi: np.ndarray) -> np.ndarray:
        """Apply the nonlocal interaction via convolution for multi-dimensional fields"""
        result = psi.copy()
        
        # Apply convolution along each dimension separately
        for dim in range(self.field_ndim):
            # For higher dimensions, need to use scipy's convolve or create a kernel tensor
            if self.field_ndim == 1:
                # 1D case: simple convolution
                pad_width = len(self.kernels[0]) // 2
                padded_psi = np.pad(result, pad_width, mode='constant')
                result_new = np.zeros_like(result)
                
                for i in range(len(result)):
                    result_new[i] = np.sum(padded_psi[i:i+len(self.kernels[0])] * self.kernels[0])
                
                result = result_new
            else:
                # Multi-dimensional case: use scipy's convolve function
                # Create a kernel of appropriate dimensionality
                kernel_shape = [1] * self.field_ndim
                kernel_shape[dim] = len(self.kernels[dim])
                kernel_nd = self.kernels[dim].reshape(kernel_shape)
                
                # Apply convolution with proper mode
                result = convolve(result, kernel_nd, mode='constant', cval=0.0)
        
        return result
    
    def _potential_gradient(self, psi: np.ndarray) -> np.ndarray:
        """Calculate gradient of the potential V(ψ) = -α/2 ψ² + β/4 ψ⁴"""
        return -self.alpha * psi + self.beta * psi**3
    
    def _diffusion_term(self, psi: np.ndarray) -> np.ndarray:
        """Calculate the diffusion term ∇·[D(ψ)∇ψ] for multi-dimensional fields"""
        # Initialize result
        laplacian = np.zeros_like(psi)
        
        # Apply Laplacian operator in each dimension
        for dim in range(self.field_ndim):
            # Create slices for finite difference calculation
            slices_before = [slice(None)] * self.field_ndim
            slices_center = [slice(None)] * self.field_ndim
            slices_after = [slice(None)] * self.field_ndim
            
            # Adjust slices for the current dimension
            slices_before[dim] = slice(0, -2)
            slices_center[dim] = slice(1, -1)
            slices_after[dim] = slice(2, None)
            
            # Compute second derivative along current dimension
            # Handle boundaries with padding
            padded = np.pad(psi, [(1, 1) if i == dim else (0, 0) for i in range(self.field_ndim)], mode='edge')
            pad_before = tuple(slice(0, -2) if i == dim else slice(None) for i in range(self.field_ndim))
            pad_center = tuple(slice(1, -1) if i == dim else slice(None) for i in range(self.field_ndim))
            pad_after = tuple(slice(2, None) if i == dim else slice(None) for i in range(self.field_ndim))
            
            # Add the second derivative term to the laplacian
            laplacian += padded[pad_before] + padded[pad_after] - 2 * padded[pad_center]
        
        return self.D * laplacian
    
    def evolve_field(self, steps: int = 1) -> None:
        """
        Evolve the field according to the IRE equation for multiple time steps
        
        Args:
            steps: Number of evolution steps to perform
        """
        for _ in range(steps):
            # Calculate right-hand side terms
            diffusion = self._diffusion_term(self.psi)
            potential = self._potential_gradient(self.psi)
            nonlocal_term = self._convolve_with_kernels(self.psi)
            
            # Update velocity (first time derivative)
            accel = diffusion - potential - nonlocal_term - self.gamma * self.psi_t
            self.psi_t += self.dt * accel
            
            # Update field using velocity
            self.psi += self.dt * self.psi_t
            
            # Record metrics
            self.evolution_steps += 1
            if self.evolution_steps % 10 == 0:  # Record every 10 steps
                self.energy_history.append(np.sum(self.psi**2))
                self.amplitude_history.append(np.mean(np.abs(self.psi)))
    
    def _initialize_projector(self):
        """Initialize the projector for mapping embeddings to field coordinates"""
        if len(self.stored_embeddings) < 10:
            return False  # Not enough data
            
        # Convert to numpy array with proper shape
        embeddings_array = np.array(self.stored_embeddings)
        if len(embeddings_array.shape) != 2:
            print(f"Warning: Unexpected embeddings shape: {embeddings_array.shape}")
            return False
        
        try:
            if self.projection_method == 'pca':
                # Initialize PCA to project embeddings to field dimensionality
                self.projector = PCA(n_components=min(self.field_ndim, embeddings_array.shape[0], embeddings_array.shape[1]))
                self.projector.fit(embeddings_array)
                self.has_enough_data_for_projection = True
                
            elif self.projection_method == 'umap':
                # Use UMAP for nonlinear projection (better preserves semantic relationships)
                if not self.has_enough_data_for_projection:  # Only initialize once
                    n_neighbors = min(15, len(embeddings_array)-1)
                    self.projector = umap.UMAP(n_components=min(self.field_ndim, embeddings_array.shape[1]),
                                              metric='cosine', 
                                              min_dist=0.1,
                                              n_neighbors=max(2, n_neighbors))
                    self.projector.fit(embeddings_array)
                    self.has_enough_data_for_projection = True
                    
            elif self.projection_method == 'direct':
                # For direct mapping, we don't need a projector - we'll use modulo
                self.has_enough_data_for_projection = True
            
            return self.has_enough_data_for_projection
        except Exception as e:
            print(f"Error initializing projector: {e}")
            self.has_enough_data_for_projection = False
            return False
    
    def _project_embedding_to_field(self, embedding: np.ndarray) -> Tuple:
        """Project a high-dimensional embedding to field coordinates"""
        # Store embedding for potential re-fitting
        self.stored_embeddings.append(embedding.copy())
        
        # Initialize projector if needed and possible
        if not self.has_enough_data_for_projection:
            self._initialize_projector()
        
        try:
            if self.projection_method == 'pca' and self.has_enough_data_for_projection:
                # Use fitted PCA to project
                coords_normalized = self.projector.transform([embedding])[0]
                
                # Scale to field dimensions
                coords = []
                for i, dim_size in enumerate(self.field_dims):
                    if i < len(coords_normalized):  # Ensure we don't go out of bounds
                        # Scale normalized coordinate (-4 to 4) to field dimensions with padding
                        c = int((coords_normalized[i] + 4) / 8 * (dim_size - 20) + 10)
                        coords.append(np.clip(c, 0, dim_size-1))
                    else:
                        # Add default position for any missing dimensions
                        coords.append(dim_size // 2)
                    
                return tuple(coords)
                
            elif self.projection_method == 'umap' and self.has_enough_data_for_projection:
                try:
                    # Use fitted UMAP to project
                    coords_normalized = self.projector.transform([embedding])[0]
                    
                    # Scale to field dimensions
                    coords = []
                    min_vals = self.projector.embedding_.min(axis=0)
                    max_vals = self.projector.embedding_.max(axis=0)
                    
                    for i, dim_size in enumerate(self.field_dims):
                        if i < len(coords_normalized):  # Ensure we don't go out of bounds
                            # Avoid division by zero
                            denom = (max_vals[i] - min_vals[i])
                            if abs(denom) < 1e-10:
                                c = dim_size // 2  # Default to middle if range is too small
                            else:
                                # Scale normalized coordinate to field dimensions with padding
                                c = int((coords_normalized[i] - min_vals[i]) / denom * 
                                      (dim_size - 20) + 10)
                            coords.append(np.clip(c, 0, dim_size-1))
                        else:
                            # Add default position for any missing dimensions
                            coords.append(dim_size // 2)
                        
                    return tuple(coords)
                except Exception as e:
                    print(f"UMAP projection error: {e}")
                    # Fall back to direct mapping
                    return self._direct_mapping(embedding)
            else:
                # Fallback or direct mapping
                return self._direct_mapping(embedding)
        except Exception as e:
            print(f"Projection error: {e}")
            # Final fallback
            return self._direct_mapping(embedding)
    
    def _direct_mapping(self, embedding: np.ndarray) -> Tuple:
        """Direct hash-based mapping as fallback method"""
        hash_val = int(hashlib.sha256(embedding.tobytes()).hexdigest(), 16)
        
        coords = []
        for dim_size in self.field_dims:
            # Use a different part of the hash for each dimension
            hash_val, coord = divmod(hash_val, dim_size)
            coords.append(coord)
            
        return tuple(coords)
    
    def add_memory_input(self, text: str, embedding: np.ndarray, strength: float = 1.0) -> int:
        """
        Add a new memory item to the field
        
        Args:
            text: The text content of the memory
            embedding: Semantic embedding of the text (used for positioning)
            strength: Strength of the memory imprint
            
        Returns:
            coordinate_id: The ID assigned to this memory coordinate
        """
        # Project high-dimensional embedding to field coordinates
        coords = self._project_embedding_to_field(embedding)
        
        # Store the semantic coordinates and embedding for later retrieval
        coordinate_id = self.next_coordinate_id
        self.next_coordinate_id += 1
        
        self.semantic_coordinates[coordinate_id] = {
            'text': text,
            'embedding': embedding,
            'coords': coords,
            'time_added': time.time(),
            'field_value': 0.0  # Will update this after adding to field
        }
        
        # Create a multi-dimensional Gaussian bump at the assigned position
        memory_input = np.zeros(self.field_dims)
        
        # Construct indices for the full grid
        grid_indices = np.meshgrid(*[np.arange(dim_size) for dim_size in self.field_dims], indexing='ij')
        
        # Calculate distance from each point to the memory position
        squared_dist = np.zeros(self.field_dims)
        for i, idx in enumerate(grid_indices):
            squared_dist += (idx - coords[i])**2
        
        # Create Gaussian bump
        width = max(4, min(10, min(self.field_dims) / 8))  # Adaptive width based on field size
        memory_input = strength * np.exp(-squared_dist / (2 * width**2))
        
        # Add the input to the field (creates a disturbance)
        self.psi += memory_input
        
        # Update the field value at this coordinate
        self.semantic_coordinates[coordinate_id]['field_value'] = float(self.psi[coords])
        
        return coordinate_id
    
    def query_field(self, embedding: np.ndarray, k: int = 3, method: str = 'semantic') -> List[Dict[str, Any]]:
        """
        Query the field to retrieve relevant memories
        
        Args:
            embedding: Semantic embedding of the query
            k: Number of memories to retrieve
            method: Query method - 'semantic' (direct embedding comparison) or 'field' (field-based)
            
        Returns:
            List of memory items with their importance weights
        """
        if method == 'semantic' or not self.has_enough_data_for_projection:
            # Direct semantic comparison without using field
            results = []
            
            for coord_id, data in self.semantic_coordinates.items():
                similarity = 1 - cosine(embedding, data['embedding'])
                field_value = float(self.psi[data['coords']])
                
                results.append({
                    'text': data['text'],
                    'importance': similarity * (0.5 + 0.5 * field_value),  # Weighted by field value
                    'similarity': similarity,
                    'field_value': field_value,
                    'coordinate_id': coord_id,
                    'time_added': data['time_added']
                })
            
            # Sort by importance and take top k
            results.sort(key=lambda x: x['importance'], reverse=True)
            return results[:k]
        
        else:
            # Field-based query using resonance
            # Project query embedding to field coordinates
            query_coords = self._project_embedding_to_field(embedding)
            
            # Create a temporary field perturbation at query position
            temp_field = np.zeros(self.field_dims)
            
            # Construct indices for the full grid
            grid_indices = np.meshgrid(*[np.arange(dim_size) for dim_size in self.field_dims], indexing='ij')
            
            # Calculate distance from each point to the query position
            squared_dist = np.zeros(self.field_dims)
            for i, idx in enumerate(grid_indices):
                squared_dist += (idx - query_coords[i])**2
            
            # Create temporary Gaussian probe
            width = max(4, min(10, min(self.field_dims) / 8))
            temp_field = np.exp(-squared_dist / (2 * width**2))
            
            # Calculate interaction with current field
            # Higher values indicate stronger resonance
            resonance = self.psi * temp_field
            
            # Find positions with highest resonance values
            flat_resonance = resonance.flatten()
            peak_indices = np.argsort(flat_resonance)[-k*2:]  # Get extra peaks to filter later
            
            # Convert flat indices back to multi-dimensional coordinates
            peak_coords = []
            for flat_idx in peak_indices:
                coords = np.unravel_index(flat_idx, self.field_dims)
                peak_coords.append(coords)
            
            # Find the stored memories closest to these resonance peaks
            results = []
            
            for peak_coord in peak_coords:
                # Find closest memory
                closest_id = None
                min_distance = float('inf')
                
                for coord_id, data in self.semantic_coordinates.items():
                    # Calculate squared distance between peak and memory coords
                    dist = sum((c1 - c2)**2 for c1, c2 in zip(peak_coord, data['coords']))
                    
                    if dist < min_distance:
                        min_distance = dist
                        closest_id = coord_id
                
                if closest_id is not None:
                    memory = self.semantic_coordinates[closest_id]
                    importance = float(resonance[peak_coord])  # Resonance strength
                    
                    # Only add if not already in results (avoid duplicates)
                    if not any(r['coordinate_id'] == closest_id for r in results):
                        results.append({
                            'text': memory['text'],
                            'importance': importance,
                            'distance': np.sqrt(min_distance),
                            'field_value': float(self.psi[memory['coords']]),
                            'coordinate_id': closest_id,
                            'time_added': memory['time_added']
                        })
            
            # Sort by importance
            results.sort(key=lambda x: x['importance'], reverse=True)
            return results[:k]
    
    def visualize_field(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the current state of the field
        
        Args:
            save_path: Optional path to save the visualization
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if len(self.field_dims) == 1:
            # 1D visualization
            plt.figure(figsize=(12, 8))
            plt.plot(self.psi)
            plt.ylabel('Field amplitude')
            plt.xlabel('Field position')
            plt.title('IRE Field State (1D)')
            
            # Mark memory positions
            if self.semantic_coordinates:
                for coord_id, data in self.semantic_coordinates.items():
                    # Truncate long labels
                    label = data['text'][:30] + '...' if len(data['text']) > 30 else data['text']
                    coord = data['coords'][0]
                    
                    plt.axvline(x=coord, color='r', linestyle='--', alpha=0.5)
                    plt.annotate(label, (coord, self.psi[coord]), fontsize=8,
                                xytext=(5, 5), textcoords='offset points')
            
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path)
            else:
                os.makedirs('SIREN/output/visualizations', exist_ok=True)
                plt.savefig(f"SIREN/output/visualizations/field_state_1D_{timestamp}.png")
            plt.close()
            
        elif len(self.field_dims) == 2:
            # 2D visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot 1: Field amplitude heatmap
            im = axes[0].imshow(self.psi, origin='lower', cmap='viridis')
            axes[0].set_title('IRE Field State (2D)')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            fig.colorbar(im, ax=axes[0], label='Field amplitude')
            
            # Plot 2: Field gradient magnitude
            gradient_y, gradient_x = np.gradient(self.psi)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            im2 = axes[1].imshow(gradient_magnitude, origin='lower', cmap='plasma')
            axes[1].set_title('Field Gradient Magnitude')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            fig.colorbar(im2, ax=axes[1], label='Gradient magnitude')
            
            # Plot 3: Memory positions with semantic clustering
            memory_coords = np.array([data['coords'] for data in self.semantic_coordinates.values()])
            memory_texts = [data['text'] for data in self.semantic_coordinates.values()]
            memory_times = [data['time_added'] for data in self.semantic_coordinates.values()]
            
            if len(memory_coords) > 0:
                # Normalize time for coloring (newer = brighter)
                now = time.time()
                time_elapsed = np.array([now - t for t in memory_times])
                max_time = max(time_elapsed) if len(time_elapsed) > 0 else 1
                normalized_time = 1 - (time_elapsed / max_time)
                
                # Plot memory positions
                scatter = axes[2].scatter(
                    memory_coords[:, 0], memory_coords[:, 1],
                    c=normalized_time, cmap='YlOrRd', 
                    s=50, alpha=0.7, edgecolors='white'
                )
                axes[2].set_title('Memory Positions')
                axes[2].set_xlabel('X')
                axes[2].set_ylabel('Y')
                fig.colorbar(scatter, ax=axes[2], label='Recency (brighter = newer)')
                
                # Add labels for the most recent memories (to avoid cluttering)
                for i, (x, y) in enumerate(memory_coords):
                    if normalized_time[i] > 0.7:  # Only label recent memories
                        label = memory_texts[i][:20] + '...' if len(memory_texts[i]) > 20 else memory_texts[i]
                        axes[2].annotate(label, (x, y), fontsize=8, 
                                       xytext=(5, 5), textcoords='offset points',
                                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
                
                # Draw field contours on top of memory positions
                contour = axes[2].contour(self.psi, cmap='Greys', alpha=0.3, levels=5)
            
            # Set consistent limits for all plots
            for ax in axes:
                ax.set_xlim(0, self.field_dims[0]-1)
                ax.set_ylim(0, self.field_dims[1]-1)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                os.makedirs('SIREN/output/visualizations', exist_ok=True)
                plt.savefig(f"SIREN/output/visualizations/field_state_2D_{timestamp}.png")
            plt.close()
            
        elif len(self.field_dims) == 3:
            # 3D visualization - show three 2D slices through the center
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Get midpoints for each dimension
            mid_x = self.field_dims[0] // 2
            mid_y = self.field_dims[1] // 2
            mid_z = self.field_dims[2] // 2
            
            # Plot slices through the middle of each dimension
            slice_xy = self.psi[:, :, mid_z]
            slice_xz = self.psi[:, mid_y, :]
            slice_yz = self.psi[mid_x, :, :]
            
            im1 = axes[0].imshow(slice_xy, origin='lower', cmap='viridis')
            axes[0].set_title(f'XY Plane (Z={mid_z})')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            fig.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(slice_xz, origin='lower', cmap='viridis')
            axes[1].set_title(f'XZ Plane (Y={mid_y})')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Z')
            fig.colorbar(im2, ax=axes[1])
            
            im3 = axes[2].imshow(slice_yz, origin='lower', cmap='viridis')
            axes[2].set_title(f'YZ Plane (X={mid_x})')
            axes[2].set_xlabel('Y')
            axes[2].set_ylabel('Z')
            fig.colorbar(im3, ax=axes[2])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                os.makedirs('SIREN/output/visualizations', exist_ok=True)
                plt.savefig(f"SIREN/output/visualizations/field_state_3D_{timestamp}.png")
            plt.close()
        
        else:
            # Higher dimensional field - create a 2D projection for visualization
            flattened_psi = self.psi.reshape(-1)
            
            # Create a simplified 2D representation by taking the first two dimensions
            if len(self.field_dims) > 3:
                # Get the first two dimensions and collapse the rest
                reshaped_dims = (self.field_dims[0], self.field_dims[1], -1)
                # Collapse by taking the maximum along the remaining dimensions
                collapsed_psi = np.max(self.psi.reshape(reshaped_dims), axis=2)
            else:
                collapsed_psi = np.mean(self.psi, axis=tuple(range(2, len(self.field_dims))))
            
            plt.figure(figsize=(12, 10))
            plt.imshow(collapsed_psi, origin='lower', cmap='viridis')
            plt.colorbar(label='Field amplitude')
            plt.title(f'IRE Field State ({len(self.field_dims)}-D Projection)')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                os.makedirs('SIREN/output/visualizations', exist_ok=True)
                plt.savefig(f"SIREN/output/visualizations/field_state_highD_{timestamp}.png")
            plt.close()
            
        # Also save a visualization of the energy over time
        if len(self.energy_history) > 10:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(self.energy_history)
            plt.title('Field Energy Over Time')
            plt.xlabel('Evolution steps (x10)')
            plt.ylabel('Total energy')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(self.amplitude_history)
            plt.title('Average Field Amplitude')
            plt.xlabel('Evolution steps (x10)')
            plt.ylabel('Mean amplitude')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                energy_save_path = save_path.replace('.png', '_energy.png')
                plt.savefig(energy_save_path)
            else:
                os.makedirs('SIREN/output/visualizations', exist_ok=True)
                plt.savefig(f"SIREN/output/visualizations/field_energy_{timestamp}.png")
            plt.close()
    
    def save_field_metrics(self, filename: str) -> Dict[str, float]:
        """
        Calculate and save various metrics about the field
        
        Args:
            filename: Filename to save metrics
            
        Returns:
            Dictionary of computed metrics
        """
        # Define all possible metrics to ensure consistent CSV structure
        metrics = {
            'timestamp': datetime.datetime.now().isoformat(),
            'average_amplitude': float(np.mean(self.psi)),
            'max_amplitude': float(np.max(self.psi)),
            'min_amplitude': float(np.min(self.psi)),
            'std_deviation': float(np.std(self.psi)),
            'total_energy': float(np.sum(self.psi**2)),
            'num_memory_items': len(self.semantic_coordinates),
            'field_entropy': float(-np.sum(self.psi**2 * np.log(np.abs(self.psi) + 1e-10))),
            'nonzero_percentage': float(np.count_nonzero(np.abs(self.psi) > 1e-5) / self.psi.size * 100),
            'avg_memory_distance': 0.0,
            'max_memory_distance': 0.0,
            'avg_resonance': 0.0,
            'max_resonance': 0.0,
            'field_dimensions': str(self.field_dims),
            'dimension_count': len(self.field_dims),
            'diffusion_constant': self.D,
            'damping': self.gamma,
            'potential_alpha': self.alpha,
            'potential_beta': self.beta,
            'evolution_steps': self.evolution_steps
        }
        
        # Calculate metrics related to semantic positioning if we have at least 2 memories
        if len(self.semantic_coordinates) > 1:
            # Calculate average distance between memory items in field space
            mem_coords = np.array([data['coords'] for data in self.semantic_coordinates.values()])
            if len(mem_coords) > 1:
                distances = []
                for i in range(len(mem_coords)):
                    for j in range(i+1, len(mem_coords)):
                        # Euclidean distance in field space
                        dist = np.sqrt(sum((mem_coords[i][d] - mem_coords[j][d])**2 
                                         for d in range(len(mem_coords[i]))))
                        distances.append(dist)
                
                metrics['avg_memory_distance'] = float(np.mean(distances))
                metrics['max_memory_distance'] = float(np.max(distances))
            
            # Calculate average semantic similarity between memory items
            mem_embeddings = np.array([data['embedding'] for data in self.semantic_coordinates.values()])
            if len(mem_embeddings) > 1:
                # Calculate pairwise cosine similarities
                similarities = []
                for i in range(len(mem_embeddings)):
                    for j in range(i+1, len(mem_embeddings)):
                        sim = 1 - cosine(mem_embeddings[i], mem_embeddings[j])
                        similarities.append(sim)
                
                metrics['avg_semantic_similarity'] = float(np.mean(similarities))
                metrics['min_semantic_similarity'] = float(np.min(similarities))
                metrics['max_semantic_similarity'] = float(np.max(similarities))
            
            # Calculate resonance metrics - how well the field represents the semantic structure
            # Check values at memory positions
            field_values = np.array([float(self.psi[data['coords']]) 
                                    for data in self.semantic_coordinates.values()])
            
            metrics['avg_memory_field_value'] = float(np.mean(field_values))
            metrics['max_memory_field_value'] = float(np.max(field_values))
            metrics['min_memory_field_value'] = float(np.min(field_values))
        
        # Save metrics to CSV
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        metrics_df = pd.DataFrame([metrics])
        
        # Check if file exists to determine if we need to write header
        file_exists = os.path.isfile(filename)
        metrics_df.to_csv(filename, mode='a', index=False, header=not file_exists)
        
        return metrics
    
    def benchmark_memory_retrieval(self, num_queries: int = 10, k: int = 3) -> Dict[str, Any]:
        """
        Benchmark memory retrieval performance of the field against ground truth
        
        Args:
            num_queries: Number of test queries to generate
            k: Number of memories to retrieve per query
            
        Returns:
            Dictionary of benchmark metrics
        """
        if len(self.semantic_coordinates) < 5:
            return {'error': 'Not enough memory items for meaningful benchmark (need at least 5)'}
        
        results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'num_queries': num_queries,
            'num_memories': len(self.semantic_coordinates),
            'k': k,
            'field_dimensions': str(self.field_dims),
            'retrieval_method': 'field',
            'semantic_retrieval_precision': 0.0,
            'field_retrieval_precision': 0.0,
            'semantic_vs_field_agreement': 0.0,
            'avg_retrieval_time_ms': 0.0,
            'semantic_mrr': 0.0,  # Mean Reciprocal Rank
            'field_mrr': 0.0,
        }
        
        # Get all memory embeddings for comparison
        all_memory_ids = list(self.semantic_coordinates.keys())
        all_embeddings = np.array([self.semantic_coordinates[mid]['embedding'] 
                                  for mid in all_memory_ids])
        
        semantic_hits = 0
        field_hits = 0
        agreement_count = 0
        total_semantic_mrr = 0
        total_field_mrr = 0
        total_time = 0
        
        # Generate test queries
        for _ in range(num_queries):
            # Randomly select a memory item as query target
            target_idx = np.random.randint(0, len(all_memory_ids))
            target_id = all_memory_ids[target_idx]
            target_embedding = all_embeddings[target_idx]
            
            # Slightly perturb the embedding to simulate an imperfect query
            noise = np.random.normal(0, 0.1, target_embedding.shape)
            query_embedding = target_embedding + noise
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Get ground truth by direct embedding comparison
            similarities = []
            for i, embedding in enumerate(all_embeddings):
                sim = 1 - cosine(query_embedding, embedding)
                similarities.append((all_memory_ids[i], sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            ground_truth = [id for id, _ in similarities[:k]]
            
            # Query using semantic method
            start_time = time.time()
            semantic_results = self.query_field(query_embedding, k, method='semantic')
            semantic_ids = [r['coordinate_id'] for r in semantic_results]
            
            # Query using field method
            field_results = self.query_field(query_embedding, k, method='field')
            field_ids = [r['coordinate_id'] for r in field_results]
            end_time = time.time()
            
            # Calculate metrics
            total_time += (end_time - start_time) * 1000  # ms
            
            # Precision - how many of our top-k are in the ground truth
            semantic_hits += len(set(semantic_ids) & set(ground_truth))
            field_hits += len(set(field_ids) & set(ground_truth))
            
            # Agreement - how many results are common between semantic and field methods
            agreement_count += len(set(semantic_ids) & set(field_ids))
            
            # Mean Reciprocal Rank (MRR)
            # Find the rank of the target_id in both result lists
            if target_id in semantic_ids:
                rank = semantic_ids.index(target_id) + 1  # 1-indexed
                total_semantic_mrr += 1 / rank
            
            if target_id in field_ids:
                rank = field_ids.index(target_id) + 1  # 1-indexed
                total_field_mrr += 1 / rank
        
        # Calculate final metrics
        results['semantic_retrieval_precision'] = semantic_hits / (num_queries * k)
        results['field_retrieval_precision'] = field_hits / (num_queries * k)
        results['semantic_vs_field_agreement'] = agreement_count / (num_queries * k)
        results['avg_retrieval_time_ms'] = total_time / (num_queries * 2)
        results['semantic_mrr'] = total_semantic_mrr / num_queries
        results['field_mrr'] = total_field_mrr / num_queries
        
        # Save benchmark results
        os.makedirs('SIREN/output/benchmarks', exist_ok=True)
        benchmark_file = f'SIREN/output/benchmarks/memory_benchmark_{datetime.datetime.now().strftime("%Y%m%d")}.csv'
        benchmark_df = pd.DataFrame([results])
        
        # Check if file exists
        file_exists = os.path.isfile(benchmark_file)
        benchmark_df.to_csv(benchmark_file, mode='a', index=False, header=not file_exists)
        
        return results


class SIRENEnhancedLLM:
    """LLM with enhanced memory through the IRE field equation"""
    
    def __init__(self, api_url: str, model: str = "nousresearch/deephermes-3-llama-3-8b-preview"):
        """
        Initialize the SIREN enhanced LLM
        
        Args:
            api_url: URL to the model API endpoint
            model: Name of the model to use
        """
        self.api_url = api_url
        self.model = model
        
        # Initialize conversation history
        self.conversation = []
        
        # Create output directories if they don't exist
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"SIREN/output/conversations/{timestamp}"
        self.metrics_dir = f"SIREN/output/metrics/{timestamp}"
        self.csv_dir = f"SIREN/output/csv/{timestamp}"
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        
        # Set up metrics CSV files
        self.conversation_csv_path = os.path.join(self.csv_dir, "conversation.csv")
        self.metrics_csv_path = os.path.join(self.csv_dir, "metrics.csv")
        
        # Initialize the CSV files with headers
        with open(self.conversation_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'role', 'content', 'field_importance', 
                           'response_time_ms', 'tokens'])
        
        with open(self.metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'metric_name', 'value'])
        
        # Cache embeddings to avoid recomputing
        self.embedding_cache = {}
        
        # Performance tracking
        self.total_tokens = 0
        self.total_response_time = 0
        self.memfield_query_time = 0
        self.memfield_update_time = 0
        
        # Field parameter tuning record - initialize before calling _get_optimized_field_parameters()
        self.parameter_tuning_history = []
        
        # Initialize the IRE memory field with optimized parameters
        field_config = self._get_optimized_field_parameters()
        self.memory_field = IRE_Field(
            field_dims=field_config['field_dims'],
            embedding_dim=field_config['embedding_dim'],
            diffusion_constant=field_config['diffusion_constant'],
            damping=field_config['damping'],
            potential_alpha=field_config['potential_alpha'],
            potential_beta=field_config['potential_beta'],
            nonlocal_scale=field_config['nonlocal_scale'],
            projection_method=field_config['projection_method']
        )
        
        print(f"SIREN enhanced LLM initialized with {self.model}")
        print(f"Using IRE field with dimensions {field_config['field_dims']}")
    
    def _get_optimized_field_parameters(self, conversation_type: str = "general") -> Dict[str, Any]:
        """
        Get optimized field parameters based on conversation type
        
        Args:
            conversation_type: Type of conversation ('general', 'technical', 'creative', 'factual')
            
        Returns:
            Dictionary of optimized parameters
        """
        # Base parameters
        params = {
            'field_dims': (128, 128),  # 2D field by default
            'embedding_dim': 768,
            'diffusion_constant': 0.1,
            'damping': 0.8,
            'potential_alpha': 0.5,
            'potential_beta': 0.1,
            'nonlocal_scale': 100,
            'projection_method': 'pca'
        }
        
        # Optimize for different conversation types
        if conversation_type == "technical":
            # Technical conversations need more precise memory recall
            # Use higher dimensionality, less diffusion, slower damping
            params.update({
                'field_dims': (128, 128, 16),  # 3D field for more capacity
                'diffusion_constant': 0.05,    # Lower diffusion for more discrete memories
                'damping': 0.7,                # Slower decay of information
                'potential_alpha': 0.6,        # Stronger organization
                'potential_beta': 0.15,        # Stronger nonlinearity
                'projection_method': 'umap'    # Better for preserving local structure
            })
        
        elif conversation_type == "creative":
            # Creative conversations benefit from more interconnections
            # Use higher diffusion, lower damping, different potential shape
            params.update({
                'field_dims': (192, 192),      # Larger 2D field for more associations
                'diffusion_constant': 0.15,    # Higher diffusion for more connections
                'damping': 0.6,                # Slower decay to maintain creative associations
                'potential_alpha': 0.4,        # Weaker organization allows more creative patterns
                'potential_beta': 0.08,        # Less nonlinearity
                'projection_method': 'umap'    # Better for creative associations
            })
            
        elif conversation_type == "factual":
            # Factual conversations need precise recall with little interference
            # Use lower diffusion, faster damping, stronger organization
            params.update({
                'field_dims': (96, 96, 24),    # 3D field with more depth/categories
                'diffusion_constant': 0.04,    # Lower diffusion for precise memory boundaries
                'damping': 0.9,                # Faster damping to reduce interference
                'potential_alpha': 0.7,        # Stronger organization for clear fact separation
                'potential_beta': 0.2,         # Higher nonlinearity for sharper fact boundaries
                'projection_method': 'pca'     # Linear projection sufficient for facts
            })
        
        # Save parameter choice
        self.parameter_tuning_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'conversation_type': conversation_type,
            'parameters': params.copy()
        })
        
        return params
    
    def tune_field_parameters(self, **kwargs) -> None:
        """
        Manually tune the IRE field parameters
        
        Args:
            **kwargs: Parameter values to update
        """
        valid_params = [
            'diffusion_constant', 'damping', 'potential_alpha', 
            'potential_beta', 'nonlocal_scale', 'projection_method'
        ]
        
        updates = {}
        for param, value in kwargs.items():
            if param in valid_params:
                if hasattr(self.memory_field, param):
                    # Update the parameter on the field
                    setattr(self.memory_field, param, value)
                    updates[param] = value
        
        # Record parameter update
        if updates:
            self.parameter_tuning_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'conversation_type': 'manual_tuning',
                'parameters': updates
            })
            
            print(f"Updated field parameters: {updates}")
        else:
            print("No valid parameters to update")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get a semantic embedding for text using SentenceTransformer
        
        Args:
            text: The text to embed
            
        Returns:
            Vector representation of the text
        """
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Initialize the embedding model if not already done
        if not hasattr(self, 'embedding_model'):
            try:
                print("Initializing SentenceTransformer embedding model...")
                # First, try to initialize on CPU to avoid CUDA errors
                import os
                # Temporarily disable CUDA
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                self.embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device='cpu')
                print("Embedding model loaded successfully on CPU")
                
                # Now try to move to MPS if available
                if torch.backends.mps.is_available():
                    try:
                        print("Moving model to MPS...")
                        self.embedding_model.to('mps')
                        print("Embedding model successfully moved to MPS")
                    except Exception as mps_error:
                        print(f"Could not move to MPS, staying on CPU: {mps_error}")
                
            except Exception as e:
                print(f"Error loading SentenceTransformer: {e}")
                print("Falling back to hash-based embeddings")
                self.embedding_model = None
        
        # Get embedding using the model if available
        if hasattr(self, 'embedding_model') and self.embedding_model is not None:
            try:
                # Truncate extremely long text to avoid OOM errors (if needed)
                max_length = 8192  # Adjust based on your hardware capabilities
                if len(text) > max_length:
                    text = text[:max_length]
                
                # Generate embedding
                embedding = self.embedding_model.encode(text)
                
                # Ensure numpy array type and normalize
                embedding = np.array(embedding)
                if np.sum(embedding**2) > 0:  # Prevent division by zero
                    embedding = embedding / np.linalg.norm(embedding)
            except Exception as e:
                print(f"Error generating embedding: {e}")
                print("Falling back to hash-based embedding")
                embedding = self._generate_hash_embedding(text)
        else:
            # Fall back to hash-based approach if model unavailable
            embedding = self._generate_hash_embedding(text)
        
        # Cache and return
        self.embedding_cache[text] = embedding
        return embedding
    
    def _generate_hash_embedding(self, text: str) -> np.ndarray:
        """
        Generate a basic hash-based embedding as fallback method
        
        Args:
            text: The text to embed
            
        Returns:
            Pseudo-random embedding vector
        """
        # Use SHA-256 for more consistent hashing
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to a list of integer values
        hash_values = [b for b in hash_bytes]
        
        # Use the hash values to seed numpy's random generator
        np.random.seed(sum(hash_values) % (2**32 - 1))
        
        # Generate a more stable pseudo-random embedding
        embedding = np.random.randn(768)  # Simulate a 768-dim embedding
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        return embedding
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation and update the memory field
        
        Args:
            role: Role of the message sender ('system', 'user', or 'assistant')
            content: Content of the message
        """
        if not content.strip():
            print("Warning: Empty message not added")
            return
            
        start_time = time.time()
        
        # Get embedding for the message
        embedding = self._get_embedding(content)
        
        # Add to memory field with appropriate strength based on role
        # System messages have higher strength to serve as guiding principles
        strength = 2.0 if role == "system" else 1.0
        
        # Evolve field before adding new memory to allow previous memories to interact
        self.memory_field.evolve_field(steps=5)
        
        # Add to memory field
        coord_id = self.memory_field.add_memory_input(content, embedding, strength=strength)
        
        # Evolve field again to incorporate the new memory
        self.memory_field.evolve_field(steps=5)
        
        # Get the field value (importance) at this memory's position
        importance = self.memory_field.semantic_coordinates[coord_id]['field_value']
        
        # Add message to the conversation history
        self.conversation.append({
            "role": role,
            "content": content,
            "embedding": embedding,  # Store for future reference
            "time_added": time.time(),
            "importance": float(importance),
            "coordinate_id": coord_id
        })
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.conversation_csv_path), exist_ok=True)
        
        # Save message to CSV
        with open(self.conversation_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().isoformat(),
                role,
                content,
                f"{importance:.4f}",
                "",  # response_time_ms (empty for user messages)
                ""   # tokens (empty for user messages)
            ])
        
        # Log metrics about field update
        update_time = (time.time() - start_time) * 1000  # convert to ms
        self._log_metric("memory_update_time_ms", update_time)
        self._log_metric("field_energy", float(np.sum(self.memory_field.psi**2)))
        self._log_metric("field_entropy", float(-np.sum(self.memory_field.psi**2 * np.log(np.abs(self.memory_field.psi) + 1e-10))))
        self._log_metric("field_message_importance", importance)
        
        # Update tracking metrics
        self.memfield_update_time += update_time
        
        # Occasionally save field metrics
        if len(self.conversation) % 5 == 0:
            self.memory_field.save_field_metrics(f"{self.metrics_dir}/field_metrics.csv")
    
    def _get_context_enhanced_prompt(self, max_context_items: int = 5) -> List[Dict[str, str]]:
        """
        Construct a prompt enhanced with context from the IRE field
        
        Args:
            max_context_items: Maximum number of context items to include
            
        Returns:
            List of message dictionaries forming the prompt
        """
        start_time = time.time()
        
        # Always include the system message
        system_messages = [m for m in self.conversation if m["role"] == "system"]
        
        # Get the most recent message (should be from the user)
        user_messages = [m for m in self.conversation if m["role"] == "user"]
        if not user_messages:
            return system_messages
        
        current_query = user_messages[-1]
        current_query_embedding = self._get_embedding(current_query["content"])
        
        # Query IRE field to get relevant memories
        retrieved_memories = self.memory_field.query_field(
            embedding=current_query_embedding,
            k=max_context_items,
            method='field'  # Use field-based query method
        )
        
        # Extract text from memories
        memory_texts = [memory["text"] for memory in retrieved_memories]
        
        # Time how long the field query takes
        field_query_time = time.time() - start_time
        
        # Find messages corresponding to these memories
        context_messages = []
        
        for message in self.conversation:
            if message["content"] in memory_texts and message not in system_messages:
                # Check if this is not the current message (avoid including the current query as context)
                if message != current_query:
                    # Find the memory importance
                    memory_idx = memory_texts.index(message["content"])
                    importance = retrieved_memories[memory_idx]["importance"]
                    
                    # Add importance information
                    context_message = message.copy()
                    context_message["importance"] = importance
                    context_messages.append(context_message)
        
        # Sort by importance (highest first)
        context_messages.sort(key=lambda x: x.get("importance", 0), reverse=True)
        
        # Limit to max_context_items
        context_messages = context_messages[:max_context_items]
        
        # Add to performance metrics
        self.memfield_query_time += field_query_time * 1000
        
        # Always include system message + current query
        # The enhanced context goes in between
        final_prompt = []
        
        # Add system messages first
        for system_msg in system_messages:
            final_prompt.append({"role": system_msg["role"], "content": system_msg["content"]})
        
        # Add retrieved context messages
        for ctx_msg in context_messages:
            # Optionally mark context messages to distinguish them
            content = f"[CONTEXT] {ctx_msg['content']}"
            final_prompt.append({"role": ctx_msg["role"], "content": content})
        
        # Add the current query last
        final_prompt.append({"role": current_query["role"], "content": current_query["content"]})
        
        # Log metrics
        self._log_metric("context_message_count", len(final_prompt))
        self._log_metric("context_retrieval_time_ms", field_query_time * 1000)
        
        return final_prompt
    
    def _log_metric(self, name: str, value: float) -> None:
        """Log a metric to the metrics CSV file"""
        with open(self.metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().isoformat(),
                name,
                value
            ])
    
    def generate_response(self, temperature: float = 0.7, max_tokens: int = -1) -> str:
        """
        Generate a response from the model with memory field enhancement
        
        Args:
            temperature: Temperature for sampling (higher = more creative)
            max_tokens: Maximum number of tokens to generate (-1 for model default)
            
        Returns:
            The generated response text
        """
        start_time = time.time()
        
        # Get context-enhanced prompt
        enhanced_prompt = self._get_context_enhanced_prompt(max_context_items=5)
        
        # Prepare the API request
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": enhanced_prompt,
            "temperature": temperature,
        }
        
        if max_tokens > 0:
            data["max_tokens"] = max_tokens
        
        # Log the prompt construction time
        prompt_time = (time.time() - start_time) * 1000  # ms
        self._log_metric("prompt_construction_time_ms", prompt_time)
        
        try:
            # Make the API call
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            response_json = response.json()
            
            # Extract the response text
            if "choices" in response_json and len(response_json["choices"]) > 0:
                if "message" in response_json["choices"][0]:
                    response_text = response_json["choices"][0]["message"]["content"]
                else:
                    response_text = "Error: Unexpected response format from API"
            else:
                response_text = "Error: No response choices returned from API"
                
            # Record performance metrics
            response_time = time.time() - start_time
            
            # Calculate token count (approximation)
            token_count = len(response_text.split())
            
            # Log these metrics
            self._log_metric("response_time_ms", response_time * 1000)
            self._log_metric("response_tokens", token_count)
            
            # Get embedding for the response
            embedding = self._get_embedding(response_text)
            
            # Evolve field before adding new memory
            self.memory_field.evolve_field(steps=5)
            
            # Add to memory field
            coord_id = self.memory_field.add_memory_input(response_text, embedding, strength=1.0)
            
            # Evolve field again to incorporate the new memory
            self.memory_field.evolve_field(steps=5)
            
            # Get the field value (importance) at this memory's position
            importance = self.memory_field.semantic_coordinates[coord_id]['field_value']
            
            # Add response to conversation history
            self.conversation.append({
                "role": "assistant",
                "content": response_text,
                "embedding": embedding,
                "time_added": time.time(),
                "importance": float(importance),
                "coordinate_id": coord_id
            })
            
            # Save to CSV
            with open(self.conversation_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.datetime.now().isoformat(),
                    "assistant",
                    response_text,
                    f"{importance:.4f}",
                    f"{response_time * 1000:.2f}",  # ms
                    token_count
                ])
            
            # Update performance tracking
            self.total_tokens += token_count
            self.total_response_time += response_time
            
            # Occasionally save field metrics and visualize
            if len(self.conversation) % 10 == 0:
                self.memory_field.save_field_metrics(f"{self.metrics_dir}/field_metrics.csv")
                self.memory_field.visualize_field(f"{self.metrics_dir}/field_state_{len(self.conversation)}.png")
            
            return response_text
            
        except Exception as e:
            error_message = f"Error: Failed to get response from API: {str(e)}"
            print(error_message)
            
            # Log the error
            self._log_metric("api_error", 1.0)
            
            # Return a fallback response
            return f"I'm sorry, I encountered an error while processing your request: {str(e)}"
    
    def visualize_memory_field(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the current state of the memory field
        
        Args:
            save_path: Optional path to save the visualization
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_path is None:
            save_path = f"SIREN/output/visualizations/memory_field_{timestamp}.png"
        
        self.memory_field.visualize_field(save_path)
    
    def print_conversation_with_importances(self) -> None:
        """
        Print the conversation messages with their importance values from the field
        """
        print("\n=== CONVERSATION WITH IRE FIELD IMPORTANCE ===")
        
        importances = []
        roles = []
        contents = []
        
        for i, message in enumerate(self.conversation):
            # Get the field value (importance) at this message's position
            importance = "N/A"
            for coord, mem in self.memory_field.semantic_coordinates.items():
                if mem['text'] == message['content']:
                    # Fix: Extract scalar value from numpy array or handle array
                    psi_value = self.memory_field.psi[coord]
                    if isinstance(psi_value, np.ndarray):
                        # If it's an array, take the mean or first element
                        psi_value = float(np.mean(psi_value))
                    else:
                        # Otherwise convert to float
                        psi_value = float(psi_value)
                    importance = f"{psi_value:.4f}"
                    importances.append(psi_value)
                    break
            
            if importance == "N/A":
                importances.append(0.0)
                
            roles.append(message['role'])
            contents.append(message['content'][:50] + "..." if len(message['content']) > 50 else message['content'])
            
            print(f"[{message['role']}] (Field: {importance})")
            print(f"{message['content']}\n")
        
        # Create a visualization of message importances
        plt.figure(figsize=(12, 6))
        
        # Color by role
        role_colors = {'system': 'blue', 'user': 'green', 'assistant': 'red'}
        colors = [role_colors.get(role, 'gray') for role in roles]
        
        plt.bar(range(len(importances)), importances, color=colors)
        plt.xlabel('Message Index')
        plt.ylabel('Field Importance')
        plt.title('Message Importance in IRE Field')
        
        # Add role labels
        plt.xticks(range(len(roles)), [f"{i}:{r[:3]}" for i, r in enumerate(roles)], rotation=45)
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=r) for r, c in role_colors.items()]
        plt.legend(handles=legend_elements)
        
        # Save the visualization
        plt.tight_layout()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"SIREN/output/visualizations/message_importances_{timestamp}.png")
        plt.close()
    
    def generate_performance_report(self) -> None:
        """Generate detailed performance metrics and visualizations"""
        # Create a timestamp for the report
        report_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load data from CSV files with error handling
        try:
            conversation_df = pd.read_csv(self.conversation_csv_path)
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"Warning: Could not read conversation CSV file: {str(e)}")
            conversation_df = pd.DataFrame()
            
        try:
            metrics_df = pd.read_csv(self.metrics_csv_path)
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"Warning: Could not read metrics CSV file: {str(e)}")
            metrics_df = pd.DataFrame()
        
        # Create output directories if they don't exist
        os.makedirs('SIREN/output/metrics', exist_ok=True)
        
        # Generate various visualizations
        
        # 1. Field importance over time (only if we have data)
        if not conversation_df.empty and 'field_importance' in conversation_df.columns and 'role' in conversation_df.columns:
            plt.figure(figsize=(12, 6))
            sns.lineplot(x=range(len(conversation_df)), y='field_importance', hue='role', data=conversation_df)
            plt.title('Field Importance Over Conversation')
            plt.xlabel('Message Index')
            plt.ylabel('Field Importance')
            plt.savefig(f"SIREN/output/metrics/field_importance_time_{report_timestamp}.png")
            plt.close()
        
        # 2. Response times (only if we have data)
        if not metrics_df.empty and 'metric_name' in metrics_df.columns:
            response_times = metrics_df[metrics_df['metric_name'] == 'response_time_ms']
            if not response_times.empty and 'value' in response_times.columns:
                plt.figure(figsize=(12, 6))
                sns.lineplot(x=range(len(response_times)), y='value', data=response_times)
                plt.title('Response Time Over Conversations')
                plt.xlabel('Response Index')
                plt.ylabel('Response Time (ms)')
                plt.savefig(f"SIREN/output/metrics/response_times_{report_timestamp}.png")
                plt.close()
        
        # 3. Context size vs response time (only if we have both metrics)
        if not metrics_df.empty and 'metric_name' in metrics_df.columns:
            context_sizes = metrics_df[metrics_df['metric_name'] == 'context_size']
            response_times = metrics_df[metrics_df['metric_name'] == 'response_time_ms']
            
            if (not response_times.empty and 'value' in response_times.columns and 
                not context_sizes.empty and 'value' in context_sizes.columns and
                len(context_sizes) == len(response_times)):
                plt.figure(figsize=(12, 6))
                plt.scatter(context_sizes['value'].values, response_times['value'].values)
                plt.title('Context Size vs Response Time')
                plt.xlabel('Context Size (number of messages)')
                plt.ylabel('Response Time (ms)')
                plt.savefig(f"SIREN/output/metrics/context_vs_response_{report_timestamp}.png")
                plt.close()
        
        # 4. Field metrics over time (only if we have data)
        if not conversation_df.empty:
            metrics_to_plot = [col for col in ['average_amplitude', 'max_amplitude', 'field_entropy', 'num_memory_items'] 
                              if col in conversation_df.columns]
            
            for metric in metrics_to_plot:
                plt.figure(figsize=(12, 6))
                sns.lineplot(x=range(len(conversation_df)), y=metric, data=conversation_df)
                plt.title(f'{metric.replace("_", " ").title()} Over Time')
                plt.xlabel('Time Step')
                plt.ylabel(metric.replace("_", " ").title())
                plt.savefig(f"SIREN/output/metrics/{metric}_{report_timestamp}.png")
                plt.close()
        
        # Generate summary report with error handling
        with open(f"SIREN/output/metrics/summary_{report_timestamp}.txt", 'w') as f:
            f.write("=== SIREN PERFORMANCE SUMMARY ===\n\n")
            
            # Message stats
            f.write(f"Total messages: {len(conversation_df)}\n")
            if not conversation_df.empty and 'role' in conversation_df.columns:
                f.write(f"User messages: {len(conversation_df[conversation_df['role'] == 'user'])}\n")
                f.write(f"Assistant messages: {len(conversation_df[conversation_df['role'] == 'assistant'])}\n\n")
            
            # Response time stats
            if not metrics_df.empty and 'metric_name' in metrics_df.columns:
                response_times = metrics_df[metrics_df['metric_name'] == 'response_time_ms']
                if not response_times.empty and 'value' in response_times.columns:
                    f.write(f"Average response time: {response_times['value'].mean():.2f} ms\n")
                    f.write(f"Max response time: {response_times['value'].max():.2f} ms\n")
                    f.write(f"Min response time: {response_times['value'].min():.2f} ms\n\n")
            
            # Field stats
            if not conversation_df.empty:
                f.write(f"Field metrics (final state):\n")
                for col in conversation_df.columns:
                    if col != 'timestamp' and len(conversation_df) > 0:
                        f.write(f"- {col}: {conversation_df[col].iloc[-1]}\n")
            
            f.write("\nDetailed CSV data and visualizations saved to SIREN/output/ directory.")
        
        print(f"Performance report generated in SIREN/output/metrics/summary_{report_timestamp}.txt")


# Example usage
def run_siren_demo():
    """Run a comprehensive demonstration and evaluation of the SIREN-enhanced LLM"""
    print("Starting SIREN Memory Enhancement Evaluation Suite...")
    
    # Initialize the enhanced LLM
    llm = SIRENEnhancedLLM(api_url="http://localhost:1234/v1/chat/completions")
    
    # Create a timestamp for this evaluation session
    eval_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add a system message
    llm.add_message("system", "You are a helpful AI assistant with enhanced memory capabilities using SIREN (Signal-Intelligent Resonance Encoding Network). You can recall information from earlier in the conversation accurately.")
    
    # Create evaluation directory
    os.makedirs('SIREN/output/evaluation', exist_ok=True)
    
    # Track correct recalls for metrics
    total_recall_tests = 0
    successful_recalls = 0
    
    # Initialize evaluation results dataframe
    eval_results = []
    
    # Run a series of systematic memory tests
    
    # Test 1: Temporal Distance Recall
    print("\n=== TEST 1: TEMPORAL DISTANCE RECALL ===")
    print("Testing ability to recall information after varying numbers of intervening messages")
    
    # Plant initial information
    facts = [
        "The atomic number of gold is 79.",
        "The capital of Madagascar is Antananarivo.",
        "The average distance from Earth to Mars is 225 million kilometers."
    ]
    
    for fact in facts:
        query = f"Please remember that {fact}"
        print(f"\n=== USER QUERY ===\n{query}")
        llm.add_message("user", query)
        response = llm.generate_response()
        print(f"\n=== AI RESPONSE ===\n{response}")
    
    # Add intervening content (noise)
    noise_topics = [
        "Tell me about the process of photosynthesis.",
        "What are some popular tourist destinations in Southeast Asia?",
        "Explain how a refrigerator works.",
        "Describe the basic rules of cricket.",
        "What are some influential movements in 20th century art?",
        "How do vaccines work to prevent disease?",
        "Explain the concept of blockchain technology."
    ]
    
    for topic in noise_topics:
        print(f"\n=== USER QUERY ===\n{topic}")
        llm.add_message("user", topic)
        response = llm.generate_response()
        print(f"\n=== AI RESPONSE ===\n{response}")
    
    # Now test recall of the original facts
    for i, fact in enumerate(facts):
        # Extract the key information to test recall
        if "gold" in fact:
            query = "What is the atomic number of gold?"
            expected = "79"
        elif "Madagascar" in fact:
            query = "What is the capital of Madagascar?"
            expected = "Antananarivo"
        elif "Mars" in fact:
            query = "What is the average distance from Earth to Mars?"
            expected = "225 million kilometers"
        
        print(f"\n=== USER QUERY ===\n{query}")
        llm.add_message("user", query)
        response = llm.generate_response()
        print(f"\n=== AI RESPONSE ===\n{response}")
        
        # Evaluate recall success
        total_recall_tests += 1
        recall_success = expected.lower() in response.lower()
        successful_recalls += 1 if recall_success else 0
        
        eval_results.append({
            'test_type': 'Temporal Distance',
            'test_id': i+1,
            'query': query,
            'expected': expected,
            'distance': len(noise_topics),
            'success': recall_success,
            'response': response[:100] + "..." if len(response) > 100 else response
        })
    
    # Visualize memory field after temporal distance test
    llm.visualize_memory_field(f"SIREN/output/evaluation/memory_field_test1.png")
    
    # Test 2: Information Interference
    print("\n=== TEST 2: INFORMATION INTERFERENCE ===")
    print("Testing ability to distinguish between similar but distinct pieces of information")
    
    # Plant sets of similar information
    similar_facts = [
        {"entity": "France", "capital": "Paris", "population": "67 million", "currency": "Euro"},
        {"entity": "Germany", "capital": "Berlin", "population": "83 million", "currency": "Euro"},
        {"entity": "Japan", "capital": "Tokyo", "population": "126 million", "currency": "Yen"},
        {"entity": "Brazil", "capital": "Brasília", "population": "213 million", "currency": "Real"}
    ]
    
    for fact in similar_facts:
        query = f"Please note that {fact['entity']}'s capital is {fact['capital']}, its population is approximately {fact['population']}, and its currency is the {fact['currency']}."
        print(f"\n=== USER QUERY ===\n{query}")
        llm.add_message("user", query)
        response = llm.generate_response()
        print(f"\n=== AI RESPONSE ===\n{response}")
    
    # Add some intervening conversation
    llm.add_message("user", "Let's talk about something else. What are some popular programming languages?")
    llm.generate_response()
    
    # Now test recall with potential for confusion
    for i, fact in enumerate(similar_facts):
        # Test capital recall
        query = f"What is the capital of {fact['entity']}?"
        expected = fact['capital']
        
        print(f"\n=== USER QUERY ===\n{query}")
        llm.add_message("user", query)
        response = llm.generate_response()
        print(f"\n=== AI RESPONSE ===\n{response}")
        
        # Evaluate recall success
        total_recall_tests += 1
        recall_success = expected.lower() in response.lower()
        successful_recalls += 1 if recall_success else 0
        
        eval_results.append({
            'test_type': 'Information Interference',
            'test_id': i+1,
            'query': query,
            'expected': expected,
            'distance': 1,
            'success': recall_success,
            'response': response[:100] + "..." if len(response) > 100 else response
        })
        
        # Test population recall
        query = f"What is the population of {fact['entity']}?"
        expected = fact['population']
        
        print(f"\n=== USER QUERY ===\n{query}")
        llm.add_message("user", query)
        response = llm.generate_response()
        print(f"\n=== AI RESPONSE ===\n{response}")
        
        # Evaluate recall success
        total_recall_tests += 1
        recall_success = expected.lower() in response.lower()
        successful_recalls += 1 if recall_success else 0
        
        eval_results.append({
            'test_type': 'Information Interference',
            'test_id': i+1,
            'query': query,
            'expected': expected,
            'distance': 1,
            'success': recall_success,
            'response': response[:100] + "..." if len(response) > 100 else response
        })
    
    # Visualize memory field after interference test
    llm.visualize_memory_field(f"SIREN/output/evaluation/memory_field_test2.png")
    
    # Test 3: Complex Reasoning Chains
    print("\n=== TEST 3: COMPLEX REASONING CHAINS ===")
    print("Testing ability to recall and connect multiple related facts")
    
    # Plant interconnected information
    chain_facts = [
        "Alice is Bob's sister.",
        "Bob is married to Carol.",
        "Carol works as a doctor.",
        "David is Alice and Bob's father.",
        "David is retired from his job as a teacher."
    ]
    
    for fact in chain_facts:
        query = f"Remember that {fact}"
        print(f"\n=== USER QUERY ===\n{query}")
        llm.add_message("user", query)
        response = llm.generate_response()
        print(f"\n=== AI RESPONSE ===\n{response}")
    
    # Add some intervening conversation
    llm.add_message("user", "Let's talk about climate change for a moment. What are the main causes?")
    llm.generate_response()
    
    # Test multi-hop reasoning questions
    reasoning_questions = [
        {"query": "What is Carol's profession?", "expected": "doctor"},
        {"query": "What is Alice's relationship to Carol?", "expected": "sister-in-law"},
        {"query": "What was David's job before retirement?", "expected": "teacher"},
        {"query": "How is David related to Carol?", "expected": "father-in-law"}
    ]
    
    for i, question in enumerate(reasoning_questions):
        print(f"\n=== USER QUERY ===\n{question['query']}")
        llm.add_message("user", question['query'])
        response = llm.generate_response()
        print(f"\n=== AI RESPONSE ===\n{response}")
        
        # Evaluate recall success
        total_recall_tests += 1
        recall_success = question['expected'].lower() in response.lower()
        successful_recalls += 1 if recall_success else 0
        
        eval_results.append({
            'test_type': 'Complex Reasoning',
            'test_id': i+1,
            'query': question['query'],
            'expected': question['expected'],
            'distance': 1,
            'success': recall_success,
            'response': response[:100] + "..." if len(response) > 100 else response
        })
    
    # Visualize memory field after reasoning test
    llm.visualize_memory_field(f"SIREN/output/evaluation/memory_field_test3.png")
    
    # Test 4: Long-term Retrieval with Semantic Cues
    print("\n=== TEST 4: LONG-TERM RETRIEVAL WITH SEMANTIC CUES ===")
    print("Testing ability to retrieve information based on semantic relatedness rather than exact matches")
    
    # Plant information with semantic connections
    semantic_facts = [
        "The Mona Lisa painting is housed in the Louvre Museum.",
        "Vincent van Gogh painted 'Starry Night' in 1889.",
        "The Hoover Dam was built during the Great Depression and completed in 1936.",
        "The Great Barrier Reef is the world's largest coral reef system and is located off the coast of Australia."
    ]
    
    for fact in semantic_facts:
        query = f"Here's an interesting fact: {fact}"
        print(f"\n=== USER QUERY ===\n{query}")
        llm.add_message("user", query)
        response = llm.generate_response()
        print(f"\n=== AI RESPONSE ===\n{response}")
    
    # Add substantial intervening conversation
    distractor_topics = [
        "Can you explain how machine learning algorithms work?",
        "What are some popular cuisines around the world?",
        "How do nuclear power plants generate electricity?",
        "Tell me about the history of the Olympic Games.",
        "What are the main differences between various programming paradigms?",
        "How does the human immune system fight disease?",
        "What are some strategies for effective time management?",
        "Explain how satellites are launched into orbit."
    ]
    
    for topic in distractor_topics:
        print(f"\n=== USER QUERY ===\n{topic}")
        llm.add_message("user", topic)
        llm.generate_response()
    
    # Test semantic retrieval with indirect questions
    semantic_questions = [
        {"query": "What famous artwork is displayed in Paris?", "expected": "Mona Lisa"},
        {"query": "Which post-impressionist artist created a famous painting of the night sky?", "expected": "Vincent van Gogh"},
        {"query": "What major infrastructure project was completed in America during the 1930s?", "expected": "Hoover Dam"},
        {"query": "What natural wonder can be found near Australia?", "expected": "Great Barrier Reef"}
    ]
    
    for i, question in enumerate(semantic_questions):
        print(f"\n=== USER QUERY ===\n{question['query']}")
        llm.add_message("user", question['query'])
        response = llm.generate_response()
        print(f"\n=== AI RESPONSE ===\n{response}")
        
        # Evaluate recall success
        total_recall_tests += 1
        recall_success = question['expected'].lower() in response.lower()
        successful_recalls += 1 if recall_success else 0
        
        eval_results.append({
            'test_type': 'Semantic Retrieval',
            'test_id': i+1,
            'query': question['query'],
            'expected': question['expected'],
            'distance': len(distractor_topics),
            'success': recall_success,
            'response': response[:100] + "..." if len(response) > 100 else response
        })
    
    # Calculate success rates
    recall_rate = (successful_recalls / total_recall_tests) * 100 if total_recall_tests > 0 else 0
    
    # Create dataframe for evaluation results
    eval_df = pd.DataFrame(eval_results)
    
    # Save evaluation results
    eval_df.to_csv(f"SIREN/output/evaluation/test_results_{eval_timestamp}.csv", index=False)
    
    # Calculate success rates by test type
    success_by_type = {}
    for test_type, group in eval_df.groupby('test_type'):
        success_rate = (group['success'].sum() / len(group)) * 100
        success_by_type[test_type] = success_rate
    
    # Generate visualization of recall rates
    plt.figure(figsize=(10, 6))
    test_types = list(success_by_type.keys())
    success_rates = list(success_by_type.values())
    plt.bar(test_types, success_rates, color='blue')
    plt.xlabel('Test Type')
    plt.ylabel('Success Rate (%)')
    plt.title('Memory Recall Success by Test Type')
    plt.tight_layout()
    plt.savefig(f"SIREN/output/evaluation/recall_by_test_{eval_timestamp}.png")
    plt.close()
    
    # Print the conversation with importance values
    llm.print_conversation_with_importances()
    
    # Generate comprehensive performance report
    llm.generate_performance_report()
    
    # Generate summary report
    with open(f"SIREN/output/evaluation/summary_{eval_timestamp}.txt", 'w') as f:
        f.write("=== SIREN MEMORY ENHANCEMENT EVALUATION SUMMARY ===\n\n")
        f.write(f"Total recall tests conducted: {total_recall_tests}\n")
        f.write(f"Successful recalls: {successful_recalls}\n")
        f.write(f"Overall recall success rate: {recall_rate:.2f}%\n\n")
        
        f.write("Performance by test type:\n")
        for test_type, success_rate in success_by_type.items():
            f.write(f"- {test_type}: {success_rate:.2f}%\n")
        
        f.write("\nTest Description:\n")
        f.write("1. Temporal Distance Recall: Tests recall after numerous intervening messages\n")
        f.write("2. Information Interference: Tests ability to distinguish similar but distinct information\n")
        f.write("3. Complex Reasoning Chains: Tests recall and connection of related information\n")
        f.write("4. Semantic Retrieval: Tests ability to recall information based on semantic rather than exact matches\n\n")
        
        f.write("Detailed results and visualizations saved to SIREN/output/evaluation/ directory.")
    
    print(f"\nSIREN evaluation completed. Overall recall success rate: {recall_rate:.2f}%")
    print(f"Detailed results saved to SIREN/output/evaluation/ directory")
    
    return recall_rate


# Check for required packages
def check_dependencies():
    """Check if all required packages are available and provide instructions if not"""
    missing_packages = []
    
    # Check core packages
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        import matplotlib
    except ImportError:
        missing_packages.append("matplotlib")
    
    # Check for sentence-transformers and its dependencies
    try:
        import sentence_transformers
    except ImportError:
        missing_packages.append("sentence-transformers")
    
    # Check for einops (required by sentence-transformers)
    try:
        import einops
    except ImportError:
        missing_packages.append("einops")
    
    # Check for optional visualization packages
    try:
        import umap
    except ImportError:
        missing_packages.append("umap-learn")
    
    # Check for machine learning packages
    try:
        import sklearn
    except ImportError:
        missing_packages.append("scikit-learn")
    
    # Check for PyTorch
    try:
        import torch
    except ImportError:
        missing_packages.append("torch")
    
    if missing_packages:
        print("Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

# Run the main program if this file is executed directly
if __name__ == "__main__":
    if check_dependencies():
        run_siren_demo()
    else:
        print("\nCannot run SIREN demo due to missing dependencies.")
        print("Please install the required packages and try again.")