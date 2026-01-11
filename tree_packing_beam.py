from decimal import Decimal, getcontext

import pandas as pd
from shapely import affinity, touches
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely import get_coordinates
import warnings
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import time

# Ignore only DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Ignore only FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Decimal precision and scaling factor
getcontext().prec = 25
scale_factor = Decimal('1e2')


class ParticipantVisibleError(Exception):
    pass


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at Tip
                (Decimal('0.0') * scale_factor, tip_y * scale_factor),
                # Right side - Top Tier
                (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
                # Right side - Middle Tier
                (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
                # Right side - Bottom Tier
                (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
                # Right Trunk
                (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
                # Left Trunk
                (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Bottom Tier
                (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Middle Tier
                (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
                # Left side - Top Tier
                (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x *scale_factor ),
                                          yoff=float(self.center_y * scale_factor))



def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    For each n-tree configuration, the metric calculates the bounding square
    volume divided by n, summed across all configurations.

    This metric uses shapely v2.1.2.

    Examples
    -------
    >>> import pandas as pd
    >>> row_id_column_name = 'id'
    >>> data = [['002_0', 's-0.2', 's-0.3', 's335'], ['002_1', 's0.49', 's0.21', 's155']]
    >>> submission = pd.DataFrame(columns=['id', 'x', 'y', 'deg'], data=data)
    >>> solution = submission[['id']].copy()
    >>> score(solution, submission, row_id_column_name)
    0.877038143325...
    """

    # remove the leading 's' from submissions
    data_cols = ['x', 'y', 'deg']
    submission = submission.astype(str)
    for c in data_cols:
        if not submission[c].str.startswith('s').all():
            raise ParticipantVisibleError(f'Value(s) in column {c} found without `s` prefix.')
        submission[c] = submission[c].str[1:]

    # enforce value limits
    limit = 100
    bad_x = (submission['x'].astype(float) < -limit).any() or             (submission['x'].astype(float) > limit).any()
    bad_y = (submission['y'].astype(float) < -limit).any() or             (submission['y'].astype(float) > limit).any()
    if bad_x or bad_y:
        raise ParticipantVisibleError('x and/or y values outside the bounds of -100 to 100.')

    # grouping puzzles to score
    submission['tree_count_group'] = submission['id'].str.split('_').str[0]

    total_score = Decimal('0.0')
    for group, df_group in submission.groupby('tree_count_group'):
        num_trees = len(df_group)

        # Create tree objects from the submission values
        placed_trees = []
        for _, row in df_group.iterrows():
            placed_trees.append(ChristmasTree(row['x'], row['y'], row['deg']))

        # Check for collisions using neighborhood search
        all_polygons = [p.polygon for p in placed_trees]
        r_tree = STRtree(all_polygons)

        # Checking for collisions
        for i, poly in enumerate(all_polygons):
            indices = r_tree.query(poly)
            for index in indices:
                if index == i:  # don't check against self
                    continue
                if poly.intersects(all_polygons[index]) and not poly.touches(all_polygons[index]):
                    raise ParticipantVisibleError(f'Overlapping trees in group {group}')

        # Calculate score for the group
        bounds = unary_union(all_polygons).bounds
        # Use the largest edge of the bounding rectangle to make a square boulding box
        side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])

        group_score = (Decimal(side_length_scaled) ** 2) / (scale_factor**2) / Decimal(num_trees)
        total_score += group_score

    return float(total_score)

@dataclass
class BeamState:
    """Represents a state in beam search."""
    trees: List[ChristmasTree]
    score: float
    
    def __lt__(self, other):
        return self.score < other.score


def calculate_bounding_square(trees: List[ChristmasTree]) -> float:
    """Calculate the side length of the minimum bounding square."""
    if not trees:
        return 0
    
    all_coords = [coord for tree in trees 
                  for coord in get_coordinates(tree.polygon).tolist()]
    all_x = [c[0] for c in all_coords]
    all_y = [c[1] for c in all_coords]
    
    width = max(all_x) - min(all_x)
    height = max(all_y) - min(all_y)
    
    return max(width, height)


def check_valid_placement(new_tree: ChristmasTree, existing_trees: List[ChristmasTree]) -> bool:
    """Check if new tree overlaps with any existing trees."""
    for tree in existing_trees:
        if new_tree.polygon.intersects(tree.polygon) and \
           not new_tree.polygon.touches(tree.polygon):
            return False
    return True


def generate_placement_candidates(trees: List[ChristmasTree], n_candidates: int = 20) -> List[ChristmasTree]:
    """Generate candidate positions for the next tree."""
    if not trees:
        # First tree at origin with common angles
        return [ChristmasTree(0, 0, angle) for angle in ['0', '90', '180', '270']]
    
    candidates = []
    
    # Get last tree as reference
    ref_tree = trees[-1]
    ref_x = float(ref_tree.center_x)
    ref_y = float(ref_tree.center_y)
    
    # Placement angles
    angles = [0, 90, 180, 270]
    
    # Placement strategies: distances and direction offsets
    distances = [0.41, 0.61, 0.81, 1.1]  # Added Buffer
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),  # Cardinal
        (0.707, 0.707), (0.707, -0.707),   # Diagonal
        (-0.707, 0.707), (-0.707, -0.707)
    ]
    
    for dist in distances:
        for dx, dy in directions:
            new_x = ref_x + dist * dx
            new_y = ref_y + dist * dy
            
            for angle in angles:
                candidate = ChristmasTree(str(new_x), str(new_y), str(angle))
                
                if check_valid_placement(candidate, trees):
                    candidates.append(candidate)
                    
                    if len(candidates) >= n_candidates:
                        return candidates
    
    # If we need more candidates, try compact grid pattern
    if len(candidates) < n_candidates:
        grid_step = 0.5
        for gx in np.arange(-2, 2.5, grid_step):
            for gy in np.arange(-2, 2.5, grid_step):
                for angle in angles:
                    candidate = ChristmasTree(str(gx), str(gy), str(angle))
                    
                    if check_valid_placement(candidate, trees):
                        candidates.append(candidate)
                        
                        if len(candidates) >= n_candidates:
                            return candidates
    
    return candidates[:n_candidates] if candidates else []


def beam_search_packing(n_trees: int, beam_width: int = 5, n_candidates: int = 20) -> List[ChristmasTree]:
    """
    Use beam search to find an efficient tree packing.
    
    Args:
        n_trees: Number of trees to pack
        beam_width: Number of top states to keep at each step
        n_candidates: Number of candidate positions to try per step
    
    Returns:
        List of placed trees
    """
    # Initialize beam with empty state
    beam = [BeamState(trees=[], score=0.0)]
    
    for step in range(n_trees):
        new_beam = []
        
        for state in beam:
            # Generate candidates for next tree
            candidates = generate_placement_candidates(state.trees, n_candidates)
            
            for candidate in candidates:
                new_trees = state.trees + [candidate]
                
                # Calculate score (smaller is better)
                square_size = calculate_bounding_square(new_trees)
                score = square_size * square_size / len(new_trees)
                
                new_beam.append(BeamState(trees=new_trees, score=score))
        
        # Keep only top beam_width states
        new_beam.sort(key=lambda x: x.score)
        beam = new_beam[:beam_width]
        
        if not beam:
            print(f"Warning: Beam empty at step {step}")
            break
    
    # Return best solution
    return beam[0].trees if beam else []


def save_submission(trees: List[ChristmasTree], n_trees: int, filename: str):
    """Save solution to CSV file."""
    with open(filename, 'w') as f:
        f.write("id,x,y,deg\n")
        for idx, tree in enumerate(trees):
            f.write(f"{n_trees:03d}_{idx},s{tree.center_x},s{tree.center_y},s{tree.angle}\n")


def main():
    """Main execution function."""
    print("=" * 70)
    print("Christmas Tree Packing - Beam Search Solution")
    print("=" * 70)
    
    start_time = time.time()
    all_solutions = []
    
    # Adaptive parameters based on tree count
    for n_trees in range(1, 201):
        iter_start = time.time()
        
        # Adjust beam width based on complexity
        if n_trees <= 10:
            beam_width = 3
            n_candidates = 15
        elif n_trees <= 50:
            beam_width = 4
            n_candidates = 18
        elif n_trees <= 100:
            beam_width = 5
            n_candidates = 20
        else:
            beam_width = 6
            n_candidates = 22
        
        # Solve
        trees = beam_search_packing(n_trees, beam_width, n_candidates)
        
        if trees and len(trees) == n_trees:
            square_size = calculate_bounding_square(trees)
            score = square_size * square_size / n_trees
            
            # Calculate efficiency
            tree_area = trees[0].polygon.area * n_trees
            square_area = square_size * square_size
            efficiency = (tree_area / square_area) * 100
            
            all_solutions.append({
                'n_trees': n_trees,
                'trees': trees,
                'score': score,
                'efficiency': efficiency
            })
            
            iter_time = time.time() - iter_start
            elapsed = time.time() - start_time
            
            print(f"N={n_trees:3d} | Score: {score:8.2f} | Eff: {efficiency:5.1f}% | "
                  f"Time: {iter_time:5.1f}s | Total: {elapsed/60:6.1f}m")
        else:
            print(f"N={n_trees:3d} | Failed to find valid solution")
    
    # Save all solutions
    print("\nSaving submission file...")
    with open('./results/submission.csv', 'w') as f:
        f.write("id,x,y,deg\n")
        for sol in all_solutions:
            for idx, tree in enumerate(sol['trees']):
                n = sol['n_trees']
                f.write(f"{n:03d}_{idx},s{tree.center_x},s{tree.center_y},s{tree.angle}\n")
    
    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"COMPLETE! Total time: {total_time/60:.1f} minutes")
    print(f"Solutions found: {len(all_solutions)}/200")
    if all_solutions:
        avg_eff = np.mean([s['efficiency'] for s in all_solutions])
        print(f"Average efficiency: {avg_eff:.1f}%")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()