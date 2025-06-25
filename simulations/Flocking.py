import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Flocking:
    """
    2D Flocking Simulation using the classic Boids algorithm.
    
    Implements three fundamental rules:
    1. Separation - avoid crowding neighbors
    2. Alignment - steer towards average heading of neighbors  
    3. Cohesion - steer towards average position of neighbors
    """
    
    def __init__(self, n_birds=20, world_size=10.0, dt=0.02, config=None):
        self.n_birds = n_birds
        self.world_size = world_size
        self.dt = dt
        
        # Set default parameters
        self._set_default_parameters()
        
        # Apply configuration if provided
        if config is not None:
            self.apply_config(config)
    
    def _set_default_parameters(self):
        """Set default flocking parameters."""
        # Flocking parameters
        self.separation_radius = 1.0
        self.alignment_radius = 2.5
        self.cohesion_radius = 2.5
        
        # Force weights
        self.separation_weight = 1.5
        self.alignment_weight = 1.0
        self.cohesion_weight = 1.0
        
        # Speed constraints
        self.max_speed = 2.0
        self.max_force = 0.1
        
        # Noise
        self.noise_strength = 0.05
        
        # Boundary parameters
        self.boundary_strength = 0.2
        self.boundary_margin = 1.0
    
    def apply_config(self, config):
        """Apply a configuration dictionary to update parameters.
        
        Args:
            config (dict): Dictionary of parameter names and values
        """
        for param_name, value in config.items():
            if hasattr(self, param_name):
                setattr(self, param_name, value)
            else:
                print(f"Warning: Parameter '{param_name}' not found. Ignoring.")
    
    def get_config(self):
        """Get current configuration as a dictionary.
        
        Returns:
            dict: Current parameter configuration
        """
        return {
            'separation_radius': self.separation_radius,
            'alignment_radius': self.alignment_radius,
            'cohesion_radius': self.cohesion_radius,
            'separation_weight': self.separation_weight,
            'alignment_weight': self.alignment_weight,
            'cohesion_weight': self.cohesion_weight,
            'max_speed': self.max_speed,
            'max_force': self.max_force,
            'noise_strength': self.noise_strength,
            'boundary_strength': self.boundary_strength,
            'boundary_margin': self.boundary_margin
        }
    
    def set_parameters(self, **kwargs):
        """
        Set flocking parameters easily.
        
        Available parameters:
        - separation_radius: Distance for separation behavior (default: 1.0)
        - alignment_radius: Distance for alignment behavior (default: 2.5)  
        - cohesion_radius: Distance for cohesion behavior (default: 2.5)
        - separation_weight: Strength of separation force (default: 1.5)
        - alignment_weight: Strength of alignment force (default: 1.0)
        - cohesion_weight: Strength of cohesion force (default: 1.0)
        - max_speed: Maximum bird speed (default: 2.0)
        - max_force: Maximum force magnitude (default: 0.1)
        - noise_strength: Random movement noise (default: 0.05)
        - boundary_strength: Boundary repulsion force (default: 0.2)
        - boundary_margin: Distance from edge to start boundary force (default: 1.0)
        """
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
                print(f"✓ Set {param} = {value}")
            else:
                print(f"⚠ Warning: Unknown parameter '{param}'")
    
    def get_parameters(self):
        """Get current parameter values as a dictionary."""
        return {
            'separation_radius': self.separation_radius,
            'alignment_radius': self.alignment_radius,
            'cohesion_radius': self.cohesion_radius,
            'separation_weight': self.separation_weight,
            'alignment_weight': self.alignment_weight,
            'cohesion_weight': self.cohesion_weight,
            'max_speed': self.max_speed,
            'max_force': self.max_force,
            'noise_strength': self.noise_strength,
            'boundary_strength': self.boundary_strength,
            'boundary_margin': self.boundary_margin
        }
    
    def print_parameters(self):
        """Print current parameter values in a readable format."""
        print("=== Flocking Parameters ===")
        params = self.get_parameters()
        for param, value in params.items():
            print(f"{param:18s}: {value}")
        print("=" * 28)
    
    def preset_murmuration(self):
        """Apply preset for tight murmuration (starling-like) behavior."""
        self.set_parameters(
            separation_radius=0.5,
            alignment_radius=2.0,
            cohesion_radius=3.0,
            separation_weight=2.0,
            alignment_weight=1.8,
            cohesion_weight=1.9,
            max_speed=3.0,
            noise_strength=0.03
        )
        print("🐦 Applied murmuration preset")
    
    def preset_geese_formation(self):
        """Apply preset for V-formation (geese-like) behavior."""
        self.set_parameters(
            separation_radius=1.2,
            alignment_radius=3.0,
            cohesion_radius=2.5,
            separation_weight=1.0,
            alignment_weight=2.5,
            cohesion_weight=1.2,
            max_speed=3.0,
            noise_strength=0.02
        )
        print("🦢 Applied geese V-formation preset")
    
    def preset_loose_flock(self):
        """Apply preset for loose, independent flocking behavior."""
        self.set_parameters(
            separation_radius=1.5,
            alignment_radius=2.0,
            cohesion_radius=3.0,
            separation_weight=2.0,
            alignment_weight=0.8,
            cohesion_weight=0.6,
            max_speed=2.5,
            noise_strength=0.1
        )
        print("🕊️ Applied loose flock preset")
    
    def preset_tight_school(self):
        """Apply preset for tight schooling (fish-like) behavior."""
        self.set_parameters(
            separation_radius=0.3,
            alignment_radius=1.5,
            cohesion_radius=1.8,
            separation_weight=1.8,
            alignment_weight=1.5,
            cohesion_weight=2.2,
            max_speed=2.5,
            noise_strength=0.02
        )
        print("🐟 Applied tight schooling preset")
    
    def reset_to_default(self):
        """Reset all parameters to default values."""
        self._set_default_parameters()
        print("🔄 Reset to default parameters")
    
    @classmethod
    def get_preset_configs(cls):
        """Get dictionary of preset configurations for different flocking behaviors.
        
        Returns:
            dict: Dictionary of preset configurations
        """
        return {
            'default': {
                'separation_radius': 1.0,
                'alignment_radius': 2.5,
                'cohesion_radius': 2.5,
                'separation_weight': 1.5,
                'alignment_weight': 1.0,
                'cohesion_weight': 1.0,
                'max_speed': 2.0,
                'max_force': 0.1,
                'noise_strength': 0.05,
                'boundary_strength': 0.2,
                'boundary_margin': 1.0
            },
            'tight_murmuration': {
                'separation_radius': 0.5,
                'alignment_radius': 2.0,
                'cohesion_radius': 2.0,
                'separation_weight': 1.2,
                'alignment_weight': 1.6,
                'cohesion_weight': 1.8,
                'max_speed': 4.0,
                'max_force': 0.15,
                'noise_strength': 0.03,
                'boundary_strength': 0.3,
                'boundary_margin': 0.8
            },
            'loose_flock': {
                'separation_radius': 1.5,
                'alignment_radius': 3.0,
                'cohesion_radius': 3.5,
                'separation_weight': 2.0,
                'alignment_weight': 0.8,
                'cohesion_weight': 0.5,
                'max_speed': 1.5,
                'max_force': 0.08,
                'noise_strength': 0.1,
                'boundary_strength': 0.15,
                'boundary_margin': 1.5
            },
            'v_formation': {
                'separation_radius': 1.2,
                'alignment_radius': 4.0,
                'cohesion_radius': 3.0,
                'separation_weight': 0.8,
                'alignment_weight': 2.5,
                'cohesion_weight': 1.2,
                'max_speed': 3.0,
                'max_force': 0.12,
                'noise_strength': 0.02,
                'boundary_strength': 0.25,
                'boundary_margin': 1.0
            },
            'fish_school': {
                'separation_radius': 0.3,
                'alignment_radius': 1.5,
                'cohesion_radius': 1.8,
                'separation_weight': 1.8,
                'alignment_weight': 1.4,
                'cohesion_weight': 2.2,
                'max_speed': 2.5,
                'max_force': 0.2,
                'noise_strength': 0.04,
                'boundary_strength': 0.4,
                'boundary_margin': 0.5
            },
            'chaotic_swarm': {
                'separation_radius': 0.8,
                'alignment_radius': 1.5,
                'cohesion_radius': 2.0,
                'separation_weight': 1.0,
                'alignment_weight': 0.5,
                'cohesion_weight': 0.8,
                'max_speed': 3.5,
                'max_force': 0.25,
                'noise_strength': 0.15,
                'boundary_strength': 0.1,
                'boundary_margin': 2.0
            }
        }
    
    def load_preset(self, preset_name):
        """Load a preset configuration.
        
        Args:
            preset_name (str): Name of the preset to load
        """
        presets = self.get_preset_configs()
        if preset_name in presets:
            self.apply_config(presets[preset_name])
            print(f"Loaded preset: {preset_name}")
        else:
            available = ', '.join(presets.keys())
            print(f"Preset '{preset_name}' not found. Available presets: {available}")
    
    def save_config(self, filename):
        """Save current configuration to a file.
        
        Args:
            filename (str): Path to save the configuration
        """
        import json
        config = self.get_config()
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {filename}")
    
    def load_config(self, filename):
        """Load configuration from a file.
        
        Args:
            filename (str): Path to load the configuration from
        """
        import json
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            self.apply_config(config)
            print(f"Configuration loaded from {filename}")
        except FileNotFoundError:
            print(f"Configuration file {filename} not found")
        except json.JSONDecodeError:
            print(f"Invalid JSON in configuration file {filename}")
        
    def initialize_flock(self, batch_size=1):
        """Initialize random positions and velocities for the flock."""
        # Positions in 2D
        positions = torch.rand(batch_size, self.n_birds, 2) * self.world_size - self.world_size/2
        
        # Random initial velocities
        velocities = torch.randn(batch_size, self.n_birds, 2) * 0.5
        
        # Normalize to reasonable speeds
        speeds = torch.norm(velocities, dim=-1, keepdim=True)
        velocities = velocities / (speeds + 1e-8) * (self.max_speed * 0.5)
        
        return positions, velocities
    
    def separation_force(self, positions, velocities):
        """Compute separation force to avoid crowding."""
        batch_size, n_birds, _ = positions.shape
        separation = torch.zeros_like(velocities)
        
        for i in range(n_birds):
            # Distance to all other birds
            diff = positions - positions[:, i:i+1, :]  # (batch, n_birds, 2)
            distances = torch.norm(diff, dim=-1)  # (batch, n_birds)
            
            # Find neighbors within separation radius
            neighbors = (distances < self.separation_radius) & (distances > 0)
            
            if neighbors.any():
                # Average displacement from neighbors (repulsive)
                neighbor_diff = diff[neighbors]
                if len(neighbor_diff) > 0:
                    avg_displacement = neighbor_diff.mean(dim=0)
                    # Normalize and scale
                    norm = torch.norm(avg_displacement)
                    if norm > 0:
                        separation[:, i, :] = avg_displacement / norm * self.max_force
        
        return separation * self.separation_weight
    
    def alignment_force(self, positions, velocities):
        """Compute alignment force to match neighbor velocities."""
        batch_size, n_birds, _ = positions.shape
        alignment = torch.zeros_like(velocities)
        
        for i in range(n_birds):
            # Distance to all other birds
            diff = positions - positions[:, i:i+1, :]
            distances = torch.norm(diff, dim=-1)
            
            # Find neighbors within alignment radius
            neighbors = (distances < self.alignment_radius) & (distances > 0)
            
            if neighbors.any():
                # Average velocity of neighbors
                neighbor_velocities = velocities[neighbors]
                if len(neighbor_velocities) > 0:
                    avg_velocity = neighbor_velocities.mean(dim=0)
                    # Desired velocity change
                    desired = avg_velocity - velocities[:, i, :]
                    # Limit force magnitude
                    norm = torch.norm(desired)
                    if norm > 0:
                        alignment[:, i, :] = desired / norm * min(norm, self.max_force)
        
        return alignment * self.alignment_weight
    
    def cohesion_force(self, positions, velocities):
        """Compute cohesion force to move towards center of neighbors."""
        batch_size, n_birds, _ = positions.shape
        cohesion = torch.zeros_like(velocities)
        
        for i in range(n_birds):
            # Distance to all other birds
            diff = positions - positions[:, i:i+1, :]
            distances = torch.norm(diff, dim=-1)
            
            # Find neighbors within cohesion radius
            neighbors = (distances < self.cohesion_radius) & (distances > 0)
            
            if neighbors.any():
                # Center of mass of neighbors
                neighbor_positions = positions[neighbors]
                if len(neighbor_positions) > 0:
                    center_of_mass = neighbor_positions.mean(dim=0)
                    # Desired direction towards center
                    desired = center_of_mass - positions[:, i, :]
                    # Limit force magnitude
                    norm = torch.norm(desired)
                    if norm > 0:
                        cohesion[:, i, :] = desired / norm * min(norm, self.max_force)
        
        return cohesion * self.cohesion_weight
    
    def boundary_force(self, positions):
        """Add gentle forces to keep birds within boundaries."""
        boundary_force = torch.zeros_like(positions)
        
        # Repel from boundaries using configurable parameters
        too_far = self.world_size / 2 - self.boundary_margin
        
        # X boundaries
        mask_left = positions[..., 0] < -too_far
        mask_right = positions[..., 0] > too_far
        boundary_force[..., 0][mask_left] = self.boundary_strength
        boundary_force[..., 0][mask_right] = -self.boundary_strength
        
        # Y boundaries  
        mask_bottom = positions[..., 1] < -too_far
        mask_top = positions[..., 1] > too_far
        boundary_force[..., 1][mask_bottom] = self.boundary_strength
        boundary_force[..., 1][mask_top] = -self.boundary_strength
        
        return boundary_force
    
    def update_step(self, positions, velocities):
        """Single simulation step."""
        # Compute forces
        sep_force = self.separation_force(positions, velocities)
        align_force = self.alignment_force(positions, velocities)
        coh_force = self.cohesion_force(positions, velocities)
        bound_force = self.boundary_force(positions)
        
        # Add noise
        noise = torch.randn_like(velocities) * self.noise_strength
        
        # Total acceleration
        acceleration = sep_force + align_force + coh_force + bound_force + noise
        
        # Update velocities
        new_velocities = velocities + acceleration * self.dt
        
        # Limit speed
        speeds = torch.norm(new_velocities, dim=-1, keepdim=True)
        new_velocities = new_velocities / (speeds + 1e-8) * torch.clamp(speeds, max=self.max_speed)
        
        # Update positions
        new_positions = positions + new_velocities * self.dt
        
        return new_positions, new_velocities
    
    def simulate(self, n_steps=1000, batch_size=10):
        """Run full simulation and return trajectory data."""
        positions, velocities = self.initialize_flock(batch_size)
        
        # Store trajectory
        trajectory = torch.zeros(n_steps, batch_size, self.n_birds, 4)  # pos_x, pos_y, vel_x, vel_y
        
        for t in range(n_steps):
            # Store current state
            trajectory[t, :, :, :2] = positions
            trajectory[t, :, :, 2:] = velocities
            
            # Update
            positions, velocities = self.update_step(positions, velocities)
        
        return trajectory
    
    def plot_frame(self, positions, velocities, title="Flocking Simulation"):
        """Plot a single frame of the simulation."""
        plt.figure(figsize=(10, 10))
        
        # Plot birds as arrows showing position and velocity direction
        for i in range(positions.shape[1]):
            x, y = positions[0, i, 0], positions[0, i, 1]
            vx, vy = velocities[0, i, 0], velocities[0, i, 1]
            
            plt.arrow(x, y, vx*0.3, vy*0.3, head_width=0.2, head_length=0.2, 
                     fc='blue', ec='blue', alpha=0.7)
            plt.scatter(x, y, c='red', s=30, alpha=0.8)
        
        plt.xlim(-self.world_size/2, self.world_size/2)
        plt.ylim(-self.world_size/2, self.world_size/2)
        plt.grid(True, alpha=0.3)
        plt.title(title)
        plt.axis('equal')
        plt.show()
    
    def animate_simulation(self, trajectory, filename="flocking.gif", fps=20):
        """Create an animation of the flocking simulation."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def animate(frame):
            ax.clear()
            positions = trajectory[frame, 0, :, :2]  # First batch
            velocities = trajectory[frame, 0, :, 2:]
            
            # Plot birds
            for i in range(positions.shape[0]):
                x, y = positions[i, 0], positions[i, 1]
                vx, vy = velocities[i, 0], velocities[i, 1]
                
                ax.arrow(x, y, vx*0.3, vy*0.3, head_width=0.2, head_length=0.2,
                        fc='blue', ec='blue', alpha=0.7)
                ax.scatter(x, y, c='red', s=30, alpha=0.8)
            
            ax.set_xlim(-self.world_size/2, self.world_size/2)
            ax.set_ylim(-self.world_size/2, self.world_size/2)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Flocking Simulation - Frame {frame}")
            ax.set_aspect('equal')
        
        anim = FuncAnimation(fig, animate, frames=trajectory.shape[0], 
                           interval=1000//fps, blit=False)
        anim.save(filename, writer='pillow', fps=fps)
        plt.close()
        return anim


# Example usage
if __name__ == "__main__":
    # Create simulation
    sim = Flocking(n_birds=25, world_size=12.0)
    
    # Run simulation
    trajectory = sim.simulate(n_steps=500, batch_size=5)
    
    # Plot initial and final frames
    sim.plot_frame(trajectory[0, :, :, :2], trajectory[0, :, :, 2:], "Initial State")
    sim.plot_frame(trajectory[-1, :, :, :2], trajectory[-1, :, :, 2:], "Final State")
    
    print(f"Generated flocking data shape: {trajectory.shape}")
    print("Data format: (time_steps, batch_size, n_birds, [pos_x, pos_y, vel_x, vel_y])")
