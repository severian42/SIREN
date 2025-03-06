#!/usr/bin/env python3
"""
SIREN CLI - Interactive Command Line Interface for SIREN-enhanced LLMs
Provides real-time conversation, visualization, and field parameter tuning.
"""

import os
import cmd
import json
import argparse
import datetime
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  
matplotlib.interactive(True) 
import matplotlib.pyplot as plt
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
import sys
import functools
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from siren_llm import SIRENEnhancedLLM, IRE_Field

# Load environment variables from .env file
load_dotenv()

console = Console()


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def ensure_return_to_prompt(func):
    """Decorator to ensure command methods return False (continue the command loop) unless explicitly returning something else"""
    @functools.wraps(func)
    def wrapper(self, arg):
        result = func(self, arg)
        # If the method doesn't return anything, return False to continue the command loop
        if result is None:
            return False
        return result
    return wrapper


class SIRENShell(cmd.Cmd):
    """Interactive shell for the SIREN-enhanced LLM"""

    intro = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ░██████╗██╗██████╗░███████╗███╗░░██╗  ░█████╗░██╗░░░░░██╗      ║
║   ██╔════╝██║██╔══██╗██╔════╝████╗░██║  ██╔══██╗██║░░░░░██║      ║
║   ╚█████╗░██║██████╔╝█████╗░░██╔██╗██║  ██║░░╚═╝██║░░░░░██║      ║
║   ░╚═══██╗██║██╔══██╗██╔══╝░░██║╚████║  ██║░░██╗██║░░░░░██║      ║
║   ██████╔╝██║██║░░██║███████╗██║░╚███║  ╚█████╔╝███████╗██║      ║
║   ╚═════╝░╚═╝╚═╝░░╚═╝╚══════╝╚═╝░░╚══╝  ░╚════╝░╚══════╝╚═╝      ║
║                                                                  ║
║       The Signal-Intelligent Resonance Encoding Network          ║
║     A Field-Theoretic Approach to LLM Context Management         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

Type 'help' or '?' to list commands.
Type 'chat <message>' to start talking to the AI.
    """
    prompt = "\n[USER]> "

    def __init__(self, api_url, model, field_dims=(128, 128), default_params=None):
        super().__init__()
        self.api_url = api_url
        self.model = model
        self.field_dims = field_dims
        self.default_params = default_params or {}
        
        with console.status("[green]Initializing SIREN...", spinner="dots"):
            # Initialize the SIREN-enhanced LLM
            self.llm = SIRENEnhancedLLM(
                api_url=api_url,
                model=model
            )
            
            # Apply default field parameters if provided
            if self.default_params:
                self.llm.tune_field_parameters(**self.default_params)
            
            # Add default system message
            self.llm.add_message("system", 
                """You are an AI assistant enhanced with SIREN (Signal-Intelligent Resonance Encoding Network) 
                memory capabilities. You can recall information accurately regardless of how long ago 
                it was mentioned in the conversation. Be helpful, accurate, and informative.""")
        
        console.print("[bold green]✓[/bold green] SIREN initialized successfully!")
        console.print(f"[bold]Model:[/bold] {model}")
        console.print(f"[bold]Field Dimensions:[/bold] {field_dims}")
        console.print("[bold]Field Parameters:[/bold]")
        self._print_field_params()
        
        # Session tracking
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use OUTPUT_DIR from environment if available
        self.output_dir = os.getenv("OUTPUT_DIR", "SIREN/output")
        self.conversation_file = f"{self.output_dir}/conversations/session_{self.session_id}.json"
        self.ensure_output_dirs()

    def ensure_output_dirs(self):
        """Ensure output directories exist"""
        os.makedirs(f"{self.output_dir}/conversations", exist_ok=True)
        os.makedirs(f"{self.output_dir}/visualizations", exist_ok=True)
        os.makedirs(f"{self.output_dir}/metrics", exist_ok=True)

    def _print_field_params(self):
        """Print current field parameters in a table"""
        table = Table(show_header=True)
        table.add_column("Parameter", style="bold")
        table.add_column("Value")
        
        field = self.llm.memory_field
        table.add_row("Diffusion Constant", f"{field.D}")
        table.add_row("Damping", f"{field.gamma}")
        table.add_row("Potential Alpha", f"{field.alpha}")
        table.add_row("Potential Beta", f"{field.beta}")
        table.add_row("Nonlocal Scale", f"{field.nonlocal_scale}")
        table.add_row("Projection Method", f"{field.projection_method}")
        
        console.print(table)

    def default(self, line):
        """Handle chat as the default behavior"""
        if line.strip():
            return self.do_chat(line)
        return False

    @ensure_return_to_prompt
    def do_chat(self, arg):
        """Chat with the SIREN-enhanced LLM"""
        if not arg:
            console.print("[red]Please provide a message to send.[/red]")
            return False
        
        # Display user message
        console.print("\n[bold blue]You:[/bold blue]")
        console.print(Panel(arg, expand=False))
        
        # Add user message to SIREN
        self.llm.add_message("user", arg)
        
        # Generate response with a spinner-style progress indicator
        with console.status("[green]Thinking...", spinner="dots"):
            response = self.llm.generate_response()
        
        # Display assistant response
        console.print("\n[bold green]SIREN:[/bold green]")
        console.print(Panel(Markdown(response), expand=False))
        
        # Save conversation
        self._save_conversation()
        
        # Make sure we return to the command loop
        console.print("")  # Add an empty line for clarity
        return False
    
    @ensure_return_to_prompt
    def do_clear(self, arg):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        return False

    @ensure_return_to_prompt
    def do_visualize(self, arg):
        """
        Visualize the current state of the memory field
        Usage: visualize
        """
        console.print("[bold]Generating field visualization...[/bold]")
        
        # Generate visualization
        vis_path = f"{self.output_dir}/visualizations/field_{self.session_id}_{len(self.llm.conversation)}.png"
        
        with console.status("[green]Creating visualization...", spinner="dots"):
            self.llm.visualize_memory_field(vis_path)
        
        console.print(f"[green]✓[/green] Visualization saved to {vis_path}")
        console.print("[yellow]Opening visualization in non-blocking mode...[/yellow]")
        
        # Open the saved visualization
        try:
            plt.ion()  # Turn on interactive mode
            plt.close('all')  # Close any existing plots
            
            img = plt.imread(vis_path)
            fig = plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            
            # Instead of plt.show(), use draw and flush_events to update without blocking
            fig.canvas.draw()
            plt.pause(0.001)  # Small pause to update the figure
            
            # Ensure the plot is visible but non-blocking
            plt.show(block=False)
            
            console.print("[green]✓[/green] Visualization displayed")
        except Exception as e:
            console.print(f"[red]Error displaying visualization: {e}[/red]")
        
        # Make sure we return to the command loop
        console.print("")  # Add an empty line for clarity
        return False

    @ensure_return_to_prompt
    def do_importance(self, arg):
        """
        Show importance values for each message in the conversation
        Usage: importance
        """
        console.print("[bold]Conversation with Field Importance Values:[/bold]\n")
        
        with console.status("[green]Analyzing message importance...", spinner="dots"):
            # Print conversation with importance values
            self.llm.print_conversation_with_importances()
            
            # Prepare importance data
            importances = []
            roles = []
            
            for i, message in enumerate(self.llm.conversation):
                if i == 0 and message['role'] == 'system':
                    continue  # Skip system message
                    
                importance = 0.0
                for coord, mem in self.llm.memory_field.semantic_coordinates.items():
                    if mem['text'] == message['content']:
                        importance = float(np.mean(self.llm.memory_field.psi[coord]))
                        break
                
                importances.append(importance)
                roles.append(message['role'])
        
        # Generate visualization
        console.print("\n[bold]Generating importance visualization...[/bold]")
        
        with console.status("[green]Creating visualization...", spinner="dots"):
            # Plot importance values
            plt.ion()  # Turn on interactive mode
            plt.close('all')  # Close any existing plots
            
            fig = plt.figure(figsize=(12, 6))
            colors = ['green' if r == 'assistant' else 'blue' if r == 'user' else 'gray' for r in roles]
            bars = plt.bar(range(len(importances)), importances, color=colors)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{roles[i][:1].upper()}',
                        ha='center', va='bottom', rotation=0)
            
            plt.xlabel('Message Index')
            plt.ylabel('Field Importance')
            plt.title('Message Importance in IRE Field')
            plt.tight_layout()
            
            # Save visualization
            imp_path = f"{self.output_dir}/visualizations/importance_{self.session_id}.png"
            plt.savefig(imp_path)
            
            # Draw the figure and wait for a short time
            fig.canvas.draw()
            plt.pause(0.001)
        
        # Ensure the plot is visible but non-blocking
        plt.show(block=False)
        console.print(f"[green]✓[/green] Importance visualization saved to {imp_path}")
        console.print("[green]✓[/green] Visualization displayed")
        
        # Make sure we return to the command loop
        console.print("")  # Add an empty line for clarity
        return False

    @ensure_return_to_prompt
    def do_tune(self, arg):
        """
        Tune field parameters to optimize performance
        Usage: tune [parameter] [value]
        Examples:
          tune damping 0.7
          tune diffusion 0.05
          tune potential_alpha 0.6
          
        Available parameters:
          - diffusion (diffusion constant)
          - damping (decay rate)
          - potential_alpha (organization strength)
          - potential_beta (nonlinearity)
          - nonlocal_scale (long-range interaction strength)
        """
        if not arg.strip():
            console.print("[bold]Current field parameters:[/bold]")
            self._print_field_params()
            return False
        
        parts = arg.split()
        if len(parts) != 2:
            console.print("[yellow]Usage: tune [parameter] [value][/yellow]")
            return False
        
        param, value = parts
        try:
            value = float(value)
        except ValueError:
            console.print("[red]Value must be a number[/red]")
            return False
        
        # Map the parameter name to the actual parameter
        param_map = {
            'diffusion': 'diffusion_constant',
            'damping': 'damping',
            'potential_alpha': 'potential_alpha',
            'potential_beta': 'potential_beta',
            'nonlocal': 'nonlocal_scale',
            'nonlocal_scale': 'nonlocal_scale'
        }
        
        if param not in param_map:
            console.print(f"[red]Unknown parameter: {param}[/red]")
            console.print("[yellow]Available parameters: diffusion, damping, potential_alpha, potential_beta, nonlocal_scale[/yellow]")
            return False
        
        # Update the parameter
        update_dict = {param_map[param]: value}
        self.llm.tune_field_parameters(**update_dict)
        
        console.print(f"[green]✓[/green] Updated {param} to {value}")
        console.print("[bold]New field parameters:[/bold]")
        self._print_field_params()
        
        return False

    @ensure_return_to_prompt
    def do_preset(self, arg):
        """
        Apply a preset configuration for specific conversation types
        Usage: preset <type>
        
        Available presets:
          - technical (optimized for technical/scientific discussions)
          - creative (optimized for creative conversations)
          - factual (optimized for fact-based Q&A)
          - default (reset to default parameters)
        """
        presets = {
            'technical': {
                'diffusion_constant': 0.05,
                'damping': 0.7,
                'potential_alpha': 0.6,
                'potential_beta': 0.15,
                'nonlocal_scale': 150
            },
            'creative': {
                'diffusion_constant': 0.15,
                'damping': 0.6,
                'potential_alpha': 0.4,
                'potential_beta': 0.08,
                'nonlocal_scale': 200
            },
            'factual': {
                'diffusion_constant': 0.04,
                'damping': 0.9,
                'potential_alpha': 0.7,
                'potential_beta': 0.2,
                'nonlocal_scale': 80
            },
            'default': {
                'diffusion_constant': 0.1,
                'damping': 0.8,
                'potential_alpha': 0.5,
                'potential_beta': 0.1,
                'nonlocal_scale': 100
            }
        }
        
        if not arg.strip() or arg not in presets:
            console.print("[yellow]Available presets: technical, creative, factual, default[/yellow]")
            return False
        
        # Apply the selected preset
        preset_name = arg.strip()
        self.llm.tune_field_parameters(**presets[preset_name])
        
        console.print(f"[green]✓[/green] Applied the '{preset_name}' preset")
        console.print("[bold]New field parameters:[/bold]")
        self._print_field_params()
        
        return False

    @ensure_return_to_prompt
    def do_save(self, arg):
        """
        Save the current conversation to a file
        Usage: save [filename]
        If filename is not provided, uses the default session filename
        """
        filename = arg.strip() if arg.strip() else self.conversation_file
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        if not os.path.dirname(filename):
            filename = f"{self.output_dir}/conversations/{filename}"
        
        self._save_conversation(filename)
        console.print(f"[green]✓[/green] Conversation saved to {filename}")
        
        return False
    
    def _save_conversation(self, filename=None):
        """Save the conversation to a file"""
        if filename is None:
            filename = self.conversation_file
            
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Create conversation data with metadata
        data = {
            'session_id': self.session_id,
            'model': self.model,
            'timestamp': datetime.datetime.now().isoformat(),
            'field_parameters': {
                'dimensions': self.field_dims,
                'diffusion_constant': self.llm.memory_field.D,
                'damping': self.llm.memory_field.gamma,
                'potential_alpha': self.llm.memory_field.alpha,
                'potential_beta': self.llm.memory_field.beta,
                'nonlocal_scale': self.llm.memory_field.nonlocal_scale,
                'projection_method': self.llm.memory_field.projection_method
            },
            'conversation': self.llm.conversation
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyJSONEncoder)

    @ensure_return_to_prompt
    def do_load(self, arg):
        """
        Load a conversation from a file
        Usage: load <filename>
        """
        if not arg.strip():
            console.print("[yellow]Please provide a filename[/yellow]")
            return False
            
        filename = arg.strip()
        if not filename.endswith('.json'):
            filename += '.json'
            
        if not os.path.isfile(filename):
            potential_paths = [
                filename,
                f"{self.output_dir}/conversations/{filename}",
                f"{self.output_dir}/conversations/{filename}.json"
            ]
            
            for path in potential_paths:
                if os.path.isfile(path):
                    filename = path
                    break
            else:
                console.print(f"[red]Could not find file: {filename}[/red]")
                return False
                
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            # Initialize a new LLM with the loaded parameters
            field_params = data.get('field_parameters', {})
            
            # Create a new SIREN instance with the loaded parameters
            self.llm = SIRENEnhancedLLM(
                api_url=self.api_url,
                model=data.get('model', self.model)
            )
            
            # Update field parameters if available
            if field_params:
                self.llm.tune_field_parameters(**{k: v for k, v in field_params.items() 
                                                 if k not in ['dimensions', 'projection_method']})
            
            # Load conversation and convert list embeddings back to NumPy arrays
            conversation = data.get('conversation', [])
            for message in conversation:
                if 'embedding' in message and isinstance(message['embedding'], list):
                    message['embedding'] = np.array(message['embedding'])
            
            self.llm.conversation = conversation
            
            # Update session info
            self.session_id = data.get('session_id', self.session_id)
            
            console.print(f"[green]✓[/green] Loaded conversation from {filename}")
            console.print(f"[bold]Session ID:[/bold] {self.session_id}")
            console.print(f"[bold]Messages:[/bold] {len(self.llm.conversation)}")
            console.print("[bold]Field Parameters:[/bold]")
            self._print_field_params()
            
        except Exception as e:
            console.print(f"[red]Error loading conversation: {e}[/red]")
            
        return False

    @ensure_return_to_prompt
    def do_report(self, arg):
        """
        Generate a performance report showing field metrics
        Usage: report
        """
        console.print("[bold]Generating performance report...[/bold]")
        
        with console.status("[green]Analyzing field...", spinner="dots"):
            self.llm.generate_performance_report()
        
        console.print("[green]✓[/green] Performance report generated")
        console.print("[yellow]Check the SIREN/output/metrics/ directory for detailed reports and visualizations[/yellow]")
        
        return False

    @ensure_return_to_prompt
    def do_benchmark(self, arg):
        """
        Run memory retrieval benchmark tests
        Usage: benchmark [num_queries]
        """
        try:
            num_queries = int(arg.strip()) if arg.strip() else 10
        except ValueError:
            console.print("[yellow]Usage: benchmark [num_queries][/yellow]")
            console.print("[yellow]num_queries must be an integer[/yellow]")
            return False
            
        console.print(f"[bold]Running memory retrieval benchmark with {num_queries} queries...[/bold]")
        
        with console.status("[green]Benchmarking...", spinner="dots"):
            results = self.llm.memory_field.benchmark_memory_retrieval(num_queries=num_queries)
            
        # Display results
        console.print("\n[bold]Benchmark Results:[/bold]")
        
        table = Table(show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        
        table.add_row("Success Rate", f"{results['success_rate']:.2f}%")
        table.add_row("Average Retrieval Time", f"{results['avg_time_ms']:.2f} ms")
        table.add_row("P@1", f"{results['precision_at_1']:.3f}")
        table.add_row("P@3", f"{results['precision_at_3']:.3f}")
        table.add_row("P@5", f"{results['precision_at_5']:.3f}")
        
        console.print(table)
        
        return False

    @ensure_return_to_prompt
    def do_system(self, arg):
        """
        Update the system message
        Usage: system <new system message>
        """
        if not arg.strip():
            # Display current system message
            system_msg = next((m['content'] for m in self.llm.conversation if m['role'] == 'system'), None)
            if system_msg:
                console.print("[bold]Current system message:[/bold]")
                console.print(Panel(system_msg, expand=False))
            else:
                console.print("[yellow]No system message set[/yellow]")
            return False
            
        # Update the system message
        system_idx = next((i for i, m in enumerate(self.llm.conversation) if m['role'] == 'system'), None)
        
        if system_idx is not None:
            self.llm.conversation[system_idx]['content'] = arg
        else:
            self.llm.conversation.insert(0, {"role": "system", "content": arg})
            
        console.print("[green]✓[/green] System message updated")
        
        return False

    @ensure_return_to_prompt
    def do_exit(self, arg):
        """Exit the SIREN CLI"""
        console.print("[bold]Saving conversation before exit...[/bold]")
        self._save_conversation()
        console.print("[bold green]Goodbye![/bold green]")
        return True
        
    @ensure_return_to_prompt
    def do_quit(self, arg):
        """Exit the SIREN CLI"""
        return self.do_exit(arg)

    @ensure_return_to_prompt
    def do_EOF(self, arg):
        """Exit on Ctrl+D"""
        print()  # Add newline
        return self.do_exit(arg)

    def process_plots(self):
        """Process any open matplotlib plots to ensure they're displayed but not blocking"""
        # Force matplotlib to process its events
        plt.pause(0.001)
        # Ensure we're in interactive mode
        if not plt.isinteractive():
            plt.ion()

    def postcmd(self, stop, line):
        """Method called after each command to ensure plots are processed"""
        # Process any plots
        self.process_plots()
        # Return normally
        return stop


def main():
    """Main entry point for the SIREN CLI"""
    parser = argparse.ArgumentParser(description="SIREN CLI for interacting with SIREN-enhanced LLMs")
    
    # Set up command-line arguments with environment variable defaults
    parser.add_argument("--api-url", type=str, 
                        default=os.getenv("API_URL", "http://localhost:11434/v1/chat/completions"),
                        help="LLM API endpoint URL")
    
    parser.add_argument("--model", type=str, 
                        default=os.getenv("MODEL_NAME", "nousresearch/deephermes-3-llama-3-8b-preview"),
                        help="Model identifier")
    
    parser.add_argument("--field-dims", type=str, 
                        default=os.getenv("FIELD_DIMENSIONS", "128,128"),
                        help="Field dimensions (comma-separated integers)")
    
    parser.add_argument("--output-dir", type=str,
                        default=os.getenv("OUTPUT_DIR", "SIREN/output"),
                        help="Directory for output files")
    
    args = parser.parse_args()
    
    # Parse field dimensions
    try:
        field_dims = tuple(int(dim) for dim in args.field_dims.split(","))
    except Exception:
        console.print("[red]Error parsing field dimensions. Using default (128,128).[/red]")
        field_dims = (128, 128)
    
    # Get default field parameters from environment
    default_params = {}
    if os.getenv("DIFFUSION_CONSTANT"):
        default_params["diffusion_constant"] = float(os.getenv("DIFFUSION_CONSTANT"))
    if os.getenv("DAMPING"):
        default_params["damping"] = float(os.getenv("DAMPING"))
    if os.getenv("POTENTIAL_ALPHA"):
        default_params["potential_alpha"] = float(os.getenv("POTENTIAL_ALPHA"))
    if os.getenv("POTENTIAL_BETA"):
        default_params["potential_beta"] = float(os.getenv("POTENTIAL_BETA"))
    if os.getenv("NONLOCAL_SCALE"):
        default_params["nonlocal_scale"] = float(os.getenv("NONLOCAL_SCALE"))
    
    # Override OUTPUT_DIR environment variable with command line argument
    if args.output_dir != os.getenv("OUTPUT_DIR", "SIREN/output"):
        os.environ["OUTPUT_DIR"] = args.output_dir
    
    # Start the interactive shell
    shell = SIRENShell(
        api_url=args.api_url, 
        model=args.model, 
        field_dims=field_dims,
        default_params=default_params
    )
    
    try:
        # Run the command loop
        shell.cmdloop()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\n")
        console.print("[bold yellow]Interrupted by user. Saving conversation before exit...[/bold yellow]")
        shell._save_conversation()
        console.print("[bold green]Goodbye![/bold green]")
    except Exception as e:
        # Handle any other exceptions
        console.print(f"[bold red]Error: {e}[/bold red]")
        console.print("[yellow]Attempting to save conversation...[/yellow]")
        try:
            shell._save_conversation()
            console.print("[green]Conversation saved successfully.[/green]")
        except Exception as save_error:
            console.print(f"[red]Failed to save conversation: {save_error}[/red]")


if __name__ == "__main__":
    main()
