#!/usr/bin/env python3
"""
Analyze QFlow calculation output and visualize convergence.
"""

import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def parse_qflow_output(filename):
    """Parse QFlow output file and extract energies and amplitudes."""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract number of cycles
    qflow_cycles_match = re.search(r'"qflow_cycles":\s*(\d+)', content)
    if not qflow_cycles_match:
        raise ValueError("Could not find qflow_cycles in output")
    num_cycles = int(qflow_cycles_match.group(1))
    
    # Extract number of active spaces
    total_combinations_match = re.search(r'Total combinations:\s*(\d+)', content)
    if not total_combinations_match:
        raise ValueError("Could not find total combinations in output")
    num_active_spaces = int(total_combinations_match.group(1))
    
    # Extract CCSD energy
    ccsd_energy = None
    ccsd_match = re.search(r'CCSD total energy / hartree\s*=\s*([-\d.]+)', content)
    if ccsd_match:
        ccsd_energy = float(ccsd_match.group(1))
    
    # Extract data for each cycle and active space
    data = defaultdict(lambda: defaultdict(dict))
    
    # Split by cycles
    cycle_pattern = r'Cycle (\d+) of \d+'
    cycle_matches = list(re.finditer(cycle_pattern, content))
    
    for cycle_idx in range(num_cycles):
        # Determine content range for this cycle
        if cycle_idx < len(cycle_matches):
            start_pos = cycle_matches[cycle_idx].start()
            end_pos = cycle_matches[cycle_idx + 1].start() if cycle_idx + 1 < len(cycle_matches) else len(content)
        else:
            # First cycle doesn't have "Cycle X of Y" header
            start_pos = 0
            end_pos = cycle_matches[0].start() if cycle_matches else len(content)
        
        cycle_content = content[start_pos:end_pos]
        
        # Find all NWQSim blocks in this cycle
        nwqsim_pattern = r'NWQSim Running QFLOW\s+Final Energy:\s*([-\d.]+)\s+Amplititudes:(.*?)NWQSim Finishes'
        nwqsim_matches = re.findall(nwqsim_pattern, cycle_content, re.DOTALL)
        
        for active_space_idx, (energy_str, amp_block) in enumerate(nwqsim_matches):
            if active_space_idx >= num_active_spaces:
                break
            
            energy = float(energy_str)
            
            # Parse amplitudes
            amplitudes = {}
            amp_pattern = r'\[([\d,\s]+)\]\s*->\s*([-\d.]+)'
            for amp_match in re.finditer(amp_pattern, amp_block):
                indices = tuple(map(int, amp_match.group(1).split(',')))
                value = float(amp_match.group(2))
                amplitudes[indices] = value
            
            cycle_num = cycle_idx + 1
            data[cycle_num][active_space_idx] = {
                'energy': energy,
                'amplitudes': amplitudes
            }
    
    return num_cycles, num_active_spaces, data, ccsd_energy


def analyze_convergence(data, num_cycles, num_active_spaces):
    """Analyze convergence of energies and amplitudes."""
    results = {
        'energy_convergence': defaultdict(list),
        'amplitude_max_change': defaultdict(list),
        'amplitude_avg_change': defaultdict(list),
        'amplitude_history': defaultdict(lambda: defaultdict(list))
    }
    
    for active_space in range(num_active_spaces):
        for cycle in range(1, num_cycles + 1):
            if cycle not in data or active_space not in data[cycle]:
                continue
            
            energy = data[cycle][active_space]['energy']
            amplitudes = data[cycle][active_space]['amplitudes']
            
            results['energy_convergence'][active_space].append(energy)
            
            # Track amplitude history
            for amp_idx, amp_val in amplitudes.items():
                results['amplitude_history'][active_space][amp_idx].append(amp_val)
            
            # Calculate amplitude changes from previous cycle
            if cycle > 1 and (cycle - 1) in data and active_space in data[cycle - 1]:
                prev_amps = data[cycle - 1][active_space]['amplitudes']
                changes = []
                for amp_idx in amplitudes:
                    if amp_idx in prev_amps:
                        change = abs(amplitudes[amp_idx] - prev_amps[amp_idx])
                        changes.append(change)
                
                if changes:
                    results['amplitude_max_change'][active_space].append(max(changes))
                    results['amplitude_avg_change'][active_space].append(np.mean(changes))
    
    return results


def plot_convergence(results, num_cycles, num_active_spaces, output_dir, data, ccsd_energy):
    """Generate convergence plots."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Plot 1: Energy convergence for all active spaces
    plt.figure(figsize=(12, 6))
    for active_space in range(num_active_spaces):
        energies = results['energy_convergence'][active_space]
        if energies:
            cycles = range(1, len(energies) + 1)
            plt.plot(cycles, energies, marker='o', label=f'Active Space {active_space + 1}')
    
    # Add CCSD energy line
    if ccsd_energy is not None:
        plt.axhline(y=ccsd_energy, color='red', linestyle='--', linewidth=2, label='CCSD Energy')
    
    plt.xlabel('Cycle')
    plt.ylabel('Final Energy (Hartree)')
    plt.title('Energy Convergence by Active Space')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'energy_convergence.png', dpi=300)
    plt.close()
    
    # Plot: All active space energies in order of appearance
    plt.figure(figsize=(14, 7))
    all_energies = []
    labels = []
    
    for cycle in range(1, num_cycles + 1):
        if cycle in data:
            for active_space in range(num_active_spaces):
                if active_space in data[cycle]:
                    energy = data[cycle][active_space]['energy']
                    all_energies.append(energy)
                    labels.append(f'C{cycle}-AS{active_space + 1}')
    
    if all_energies:
        x_positions = range(len(all_energies))
        plt.plot(x_positions, all_energies, marker='o', linestyle='-', linewidth=1, markersize=4)
        
        # Add CCSD energy line
        if ccsd_energy is not None:
            plt.axhline(y=ccsd_energy, color='red', linestyle='--', linewidth=2, label='CCSD Energy')
            plt.legend()
        
        plt.xlabel('Active Space Index (in order of appearance)')
        plt.ylabel('Final Energy (Hartree)')
        plt.title('All Active Space Energies (All Cycles in Order)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'all_energies_ordered.png', dpi=300)
        plt.close()
    
    # Plot 2: Maximum amplitude change
    plt.figure(figsize=(12, 6))
    for active_space in range(num_active_spaces):
        max_changes = results['amplitude_max_change'][active_space]
        if max_changes:
            cycles = range(2, len(max_changes) + 2)
            plt.plot(cycles, max_changes, marker='o', label=f'Active Space {active_space + 1}')
    
    plt.xlabel('Cycle')
    plt.ylabel('Maximum Amplitude Change')
    plt.title('Maximum Amplitude Change Between Cycles')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'amplitude_max_change.png', dpi=300)
    plt.close()
    
    # Plot 3: Average amplitude change
    plt.figure(figsize=(12, 6))
    for active_space in range(num_active_spaces):
        avg_changes = results['amplitude_avg_change'][active_space]
        if avg_changes:
            cycles = range(2, len(avg_changes) + 2)
            plt.plot(cycles, avg_changes, marker='o', label=f'Active Space {active_space + 1}')
    
    plt.xlabel('Cycle')
    plt.ylabel('Average Amplitude Change')
    plt.title('Average Amplitude Change Between Cycles')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'amplitude_avg_change.png', dpi=300)
    plt.close()
    
    # Plot: Last cycle energies vs CCSD energy
    plt.figure(figsize=(12, 6))
    last_cycle_energies = []
    active_space_ids = []
    
    if num_cycles in data:
        for active_space in range(num_active_spaces):
            if active_space in data[num_cycles]:
                energy = data[num_cycles][active_space]['energy']
                last_cycle_energies.append(energy)
                active_space_ids.append(active_space + 1)
    
    if last_cycle_energies:
        plt.scatter(active_space_ids, last_cycle_energies, s=100, alpha=0.6, label='Active Space Energies')
        
        # Add CCSD energy line
        if ccsd_energy is not None:
            plt.axhline(y=ccsd_energy, color='red', linestyle='--', linewidth=2, label='CCSD Energy')
        
        plt.xlabel('Active Space Number')
        plt.ylabel('Final Energy (Hartree)')
        plt.title(f'Last Cycle (Cycle {num_cycles}) Active Space Energies vs CCSD')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'last_cycle_vs_ccsd.png', dpi=300)
        plt.close()
    
    # NEW: Per-cycle statistics plots
    plot_per_cycle_statistics(data, num_cycles, num_active_spaces, output_path, ccsd_energy)
    
    # NEW: Plot energies only up to last complete cycle
    plot_complete_cycles_only(data, num_cycles, num_active_spaces, output_path, ccsd_energy)
    
    print(f"\nPlots saved to {output_path}/")


def plot_per_cycle_statistics(data, num_cycles, num_active_spaces, output_path, ccsd_energy):
    """Generate plots showing per-cycle statistical trends."""
    
    # Collect per-cycle statistics
    cycle_numbers = []
    highest_energies = []
    lowest_energies = []
    avg_energies = []
    median_energies = []
    energy_ranges = []
    completed_spaces = []
    max_energy_changes = []
    avg_energy_changes = []
    
    prev_cycle_energies = None
    
    for cycle in range(1, num_cycles + 1):
        cycle_analysis = analyze_cycle(data, cycle, num_active_spaces, prev_cycle_energies)
        
        if cycle_analysis is None:
            continue
        
        cycle_numbers.append(cycle)
        highest_energies.append(cycle_analysis['highest_energy'])
        lowest_energies.append(cycle_analysis['lowest_energy'])
        avg_energies.append(cycle_analysis['avg_energy'])
        median_energies.append(cycle_analysis['median_energy'])
        energy_ranges.append(cycle_analysis['highest_energy'] - cycle_analysis['lowest_energy'])
        completed_spaces.append(cycle_analysis['num_active_spaces_completed'])
        
        if 'max_energy_change' in cycle_analysis:
            max_energy_changes.append(cycle_analysis['max_energy_change'])
            avg_energy_changes.append(cycle_analysis['avg_energy_change'])
        
        prev_cycle_energies = cycle_analysis['energies']
    
    if not cycle_numbers:
        return
    
    # Plot 1: Highest and lowest energies per cycle
    plt.figure(figsize=(12, 6))
    plt.plot(cycle_numbers, highest_energies, marker='o', label='Highest Energy', color='red')
    plt.plot(cycle_numbers, lowest_energies, marker='s', label='Lowest Energy', color='blue')
    plt.plot(cycle_numbers, avg_energies, marker='^', label='Average Energy', color='green')
    plt.plot(cycle_numbers, median_energies, marker='d', label='Median Energy', color='orange')
    
    if ccsd_energy is not None:
        plt.axhline(y=ccsd_energy, color='black', linestyle='--', linewidth=2, label='CCSD Energy')
    
    plt.xlabel('Cycle')
    plt.ylabel('Energy (Hartree)')
    plt.title('Energy Statistics Per Cycle')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'per_cycle_energy_stats.png', dpi=300)
    plt.close()
    
    # Plot 2: Energy range per cycle
    plt.figure(figsize=(12, 6))
    plt.plot(cycle_numbers, energy_ranges, marker='o', color='purple')
    plt.xlabel('Cycle')
    plt.ylabel('Energy Range (Highest - Lowest) [Hartree]')
    plt.title('Energy Range Per Cycle')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'per_cycle_energy_range.png', dpi=300)
    plt.close()
    
    # Plot 3: Number of completed active spaces per cycle
    plt.figure(figsize=(12, 6))
    plt.bar(cycle_numbers, completed_spaces, alpha=0.7, color='skyblue')
    plt.axhline(y=num_active_spaces, color='red', linestyle='--', linewidth=2, 
                label=f'Total Active Spaces ({num_active_spaces})')
    plt.xlabel('Cycle')
    plt.ylabel('Number of Completed Active Spaces')
    plt.title('Active Spaces Completed Per Cycle')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path / 'per_cycle_completed_spaces.png', dpi=300)
    plt.close()
    
    # Plot 4: Energy changes from previous cycle (if available)
    if max_energy_changes:
        plt.figure(figsize=(12, 6))
        change_cycles = cycle_numbers[1:]  # Skip first cycle since no previous cycle exists
        plt.plot(change_cycles, max_energy_changes, marker='o', label='Max Energy Change', color='red')
        if avg_energy_changes:
            plt.plot(change_cycles, avg_energy_changes, marker='s', label='Avg Energy Change', color='blue')
        
        plt.xlabel('Cycle')
        plt.ylabel('Energy Change from Previous Cycle')
        plt.title('Energy Changes Between Cycles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(output_path / 'per_cycle_energy_changes.png', dpi=300)
        plt.close()


def plot_complete_cycles_only(data, num_cycles, num_active_spaces, output_path, ccsd_energy):
    """Plot all active space energies in order of appearance, but only for complete cycles."""
    
    # Find complete cycles (cycles where all active spaces finished)
    complete_cycles = []
    for cycle in range(1, num_cycles + 1):
        if cycle in data:
            # Check if all active spaces are present in this cycle
            completed_spaces = sum(1 for active_space in range(num_active_spaces) 
                                 if active_space in data[cycle])
            if completed_spaces == num_active_spaces:
                complete_cycles.append(cycle)
    
    if not complete_cycles:
        print("No complete cycles found for plotting.")
        return
    
    plt.figure(figsize=(14, 7))
    all_energies = []
    labels = []
    cycle_colors = []
    
    # Use a color map for different cycles
    cmap = plt.colormaps['tab10']
    
    for cycle in complete_cycles:
        for active_space in range(num_active_spaces):
            if active_space in data[cycle]:
                energy = data[cycle][active_space]['energy']
                all_energies.append(energy)
                labels.append(f'C{cycle}-AS{active_space + 1}')
                cycle_colors.append(cmap((cycle-1) % 10))
    
    if all_energies:
        x_positions = range(len(all_energies))
        
        # Create scatter plot with different colors for each cycle
        for i, (x, y, label) in enumerate(zip(x_positions, all_energies, labels)):
            cycle_num = int(label.split('-')[0][1:])  # Extract cycle number
            plt.scatter(x, y, color=cmap((cycle_num-1) % 10), s=50, alpha=0.7)
        
        # Also add a connecting line
        plt.plot(x_positions, all_energies, linestyle='-', linewidth=1, alpha=0.5, color='gray')
        
        # Add CCSD energy line
        if ccsd_energy is not None:
            plt.axhline(y=ccsd_energy, color='red', linestyle='--', linewidth=2, label='CCSD Energy')
        
        # Create custom legend for cycles
        legend_elements = []
        if ccsd_energy is not None:
            legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='CCSD Energy'))
        
        for cycle in complete_cycles:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=cmap((cycle-1) % 10), 
                                            markersize=8, label=f'Cycle {cycle}'))
        
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel('Active Space Index (in order of appearance)')
        plt.ylabel('Final Energy (Hartree)')
        
        if len(complete_cycles) == 1:
            plt.title(f'All Active Space Energies (Complete Cycle {complete_cycles[0]} Only)')
        else:
            plt.title(f'All Active Space Energies (Complete Cycles {complete_cycles[0]}-{complete_cycles[-1]} Only)')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'complete_cycles_only.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created complete cycles plot with {len(complete_cycles)} complete cycles: {complete_cycles}")
    else:
        print("No energy data found for complete cycles.")


def analyze_cycle(data, cycle_num, num_active_spaces, prev_cycle_data=None):
    """Analyze a specific cycle and return statistics."""
    cycle_energies = {}
    
    if cycle_num not in data:
        return None
    
    # Collect all energies from this cycle
    for active_space in range(num_active_spaces):
        if active_space in data[cycle_num]:
            energy = data[cycle_num][active_space]['energy']
            cycle_energies[active_space] = energy
    
    if not cycle_energies:
        return None
    
    # Find highest and lowest energies
    highest_space = max(cycle_energies, key=cycle_energies.get)
    lowest_space = min(cycle_energies, key=cycle_energies.get)
    
    energies_list = list(cycle_energies.values())
    avg_energy = np.mean(energies_list)
    median_energy = np.median(energies_list)
    
    analysis = {
        'energies': cycle_energies,
        'highest_energy': cycle_energies[highest_space],
        'highest_space': highest_space + 1,
        'lowest_energy': cycle_energies[lowest_space],
        'lowest_space': lowest_space + 1,
        'avg_energy': avg_energy,
        'median_energy': median_energy,
        'num_active_spaces_completed': len(cycle_energies)
    }
    
    # Calculate energy changes from previous cycle
    if prev_cycle_data and cycle_num > 1:
        max_change = 0.0
        max_change_space = None
        energy_changes = []
        
        for active_space in range(num_active_spaces):
            if (active_space in data[cycle_num] and 
                active_space in prev_cycle_data):
                current_energy = data[cycle_num][active_space]['energy']
                prev_energy = prev_cycle_data[active_space]
                change = abs(current_energy - prev_energy)
                energy_changes.append(change)
                
                if max_change_space is None or change > max_change:
                    max_change = change
                    max_change_space = active_space
        
        if max_change_space is not None:
            analysis['max_energy_change'] = max_change
            analysis['max_change_space'] = max_change_space + 1
            analysis['avg_energy_change'] = np.mean(energy_changes)
            analysis['median_energy_change'] = np.median(energy_changes)
    
    return analysis


def print_summary(data, results, num_cycles, num_active_spaces):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("QFlow Convergence Analysis Summary")
    print("="*70)
    print(f"Number of cycles: {num_cycles}")
    print(f"Number of active spaces: {num_active_spaces}")
    print()
    
    for active_space in range(num_active_spaces):
        print(f"\nActive Space {active_space + 1}:")
        print("-" * 70)
        
        energies = results['energy_convergence'][active_space]
        if len(energies) > 1:
            energy_changes = [abs(energies[i] - energies[i-1]) for i in range(1, len(energies))]
            print(f"  Final energy (last cycle): {energies[-1]:.10f} Hartree")
            print(f"  Energy change (last step): {energy_changes[-1]:.2e}")
            print(f"  Max energy change: {max(energy_changes):.2e}")
            print(f"  Avg energy change: {np.mean(energy_changes):.2e}")
        
        max_changes = results['amplitude_max_change'][active_space]
        avg_changes = results['amplitude_avg_change'][active_space]
        
        if max_changes:
            print(f"  Max amplitude change (last step): {max_changes[-1]:.2e}")
            print(f"  Avg amplitude change (last step): {avg_changes[-1]:.2e}")
            print(f"  Overall max amplitude change: {max(max_changes):.2e}")
            print(f"  Overall avg amplitude change: {np.mean(avg_changes):.2e}")
    
    # Per-cycle analysis
    print("\n" + "="*70)
    print("Per-Cycle Analysis")
    print("="*70)
    
    prev_cycle_energies = None
    
    for cycle in range(1, num_cycles + 1):
        print(f"\n--- Cycle {cycle} ---")
        
        cycle_analysis = analyze_cycle(data, cycle, num_active_spaces, prev_cycle_energies)
        
        if cycle_analysis is None:
            print("  No data available for this cycle.")
            continue
        
        print(f"  Active spaces completed: {cycle_analysis['num_active_spaces_completed']}/{num_active_spaces}")
        print(f"  Highest energy: {cycle_analysis['highest_energy']:.10f} Hartree (AS {cycle_analysis['highest_space']})")
        print(f"  Lowest energy:  {cycle_analysis['lowest_energy']:.10f} Hartree (AS {cycle_analysis['lowest_space']})")
        print(f"  Average energy: {cycle_analysis['avg_energy']:.10f} Hartree")
        print(f"  Median energy:  {cycle_analysis['median_energy']:.10f} Hartree")
        
        if 'max_energy_change' in cycle_analysis:
            print(f"  Max energy change from prev cycle: {cycle_analysis['max_energy_change']:.2e} (AS {cycle_analysis['max_change_space']})")
            print(f"  Avg energy change from prev cycle: {cycle_analysis['avg_energy_change']:.2e}")
            print(f"  Median energy change from prev cycle: {cycle_analysis['median_energy_change']:.2e}")
        
        # Store energies for next iteration
        prev_cycle_energies = cycle_analysis['energies']
    
    # Final analysis section (kept for backward compatibility)
    print("\n" + "="*70)
    print("Final Analysis (Last Cycle)")
    print("="*70)
    
    final_analysis = analyze_cycle(data, num_cycles, num_active_spaces, 
                                 data.get(num_cycles - 1, {}) if num_cycles > 1 else None)
    
    if final_analysis:
        print(f"\nHighest active space energy: {final_analysis['highest_energy']:.10f} Hartree")
        print(f"  Active Space Number: {final_analysis['highest_space']}")
        print(f"\nLowest active space energy:  {final_analysis['lowest_energy']:.10f} Hartree")
        print(f"  Active Space Number: {final_analysis['lowest_space']}")
        print(f"\nAverage of active space energies: {final_analysis['avg_energy']:.10f} Hartree")
        print(f"Median of active space energies:  {final_analysis['median_energy']:.10f} Hartree")
        
        if 'max_energy_change' in final_analysis:
            print(f"\nLargest energy change from previous cycle: {final_analysis['max_energy_change']:.2e}")
            print(f"  Active Space ID: {final_analysis['max_change_space']}")
    else:
        print("\nNo data available for the last cycle.")
    
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_qflow.py <output_file> [output_dir]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "qflow_analysis"
    
    print(f"Parsing {input_file}...")
    num_cycles, num_active_spaces, data, ccsd_energy = parse_qflow_output(input_file)
    
    print(f"Found {num_cycles} cycles and {num_active_spaces} active spaces")
    if ccsd_energy is not None:
        print(f"CCSD energy: {ccsd_energy:.10f} Hartree")
    
    print("Analyzing convergence...")
    results = analyze_convergence(data, num_cycles, num_active_spaces)
    
    print("Generating plots...")
    plot_convergence(results, num_cycles, num_active_spaces, output_dir, data, ccsd_energy)
    
    print_summary(data, results, num_cycles, num_active_spaces)
    
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main()
