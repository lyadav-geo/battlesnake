"""
Tournament-Level Battlesnake AI
================================
A competitive Battlesnake implementation featuring:
- Advanced A* pathfinding with multiple cost models
- Voronoi territory control
- Opponent modeling and prediction
- Adaptive game phase strategies
- Aggressive trapping and hunting
- Robust survival mechanisms
"""

import typing
from collections import deque, defaultdict
import heapq
import math

# Global state for tracking opponent behavior across turns
opponent_history = defaultdict(lambda: {"moves": [], "aggression": 0.5, "risk_tolerance": 0.5})


def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "Tournament Viper",
        "color": "#FF0000",
        "head": "evil",
        "tail": "sharp",
    }


def start(game_state: typing.Dict):
    pass


def end(game_state: typing.Dict):
    pass


# ============================================================================
# CORE UTILITIES
# ============================================================================

def get_next_position(pos: typing.Dict, direction: str) -> typing.Dict:
    """Calculate next position given current position and direction"""
    x, y = pos["x"], pos["y"]
    if direction == "up":
        return {"x": x, "y": y + 1}
    elif direction == "down":
        return {"x": x, "y": y - 1}
    elif direction == "left":
        return {"x": x - 1, "y": y}
    elif direction == "right":
        return {"x": x + 1, "y": y}
    return pos


def is_out_of_bounds(pos: typing.Dict, board_width: int, board_height: int) -> bool:
    """Check if position is outside board boundaries"""
    return pos["x"] < 0 or pos["x"] >= board_width or pos["y"] < 0 or pos["y"] >= board_height


def manhattan_distance(pos1: typing.Dict, pos2: typing.Dict) -> int:
    """Calculate Manhattan distance between two positions"""
    return abs(pos1["x"] - pos2["x"]) + abs(pos1["y"] - pos2["y"])


def get_all_snake_positions(game_state: typing.Dict, exclude_tails: bool = True) -> set:
    """
    Get all positions occupied by snake bodies.
    If exclude_tails=True, excludes tails that will move next turn (unless snake just ate).
    """
    positions = set()
    
    for snake in game_state["board"]["snakes"]:
        body = snake.get("body", [])
        if len(body) == 0:
            continue
        
        # Check if snake just ate (tail duplicated)
        snake_just_ate = len(body) > 1 and body[-1]["x"] == body[-2]["x"] and body[-1]["y"] == body[-2]["y"]
        
        for i, segment in enumerate(body):
            # Skip head (will move)
            if i == 0:
                continue
            
            # Skip tail unless snake just ate or we want all positions
            if i == len(body) - 1 and exclude_tails and not snake_just_ate:
                continue
            
            positions.add((segment["x"], segment["y"]))
    
    return positions


def get_directions() -> typing.List[str]:
    """Get all possible movement directions"""
    return ["up", "down", "left", "right"]


# ============================================================================
# PATHFINDING
# ============================================================================

def astar_pathfind(start: typing.Dict, goal: typing.Dict, game_state: typing.Dict, 
                   obstacles: set, allow_goal_obstacle: bool = True,
                   cost_fn=None) -> typing.Tuple[typing.Optional[int], typing.Optional[typing.List]]:
    """
    A* pathfinding with optional custom cost function.
    Returns (distance, path) or (None, None) if no path exists.
    cost_fn(current_pos, next_pos) returns additional cost for moving to next_pos.
    """
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    
    start_tuple = (start["x"], start["y"])
    goal_tuple = (goal["x"], goal["y"])
    
    # Priority queue: (f_score, g_score, position)
    open_set = []
    heapq.heappush(open_set, (0, 0, start_tuple))
    
    # Track best g_score for each position
    g_scores = {start_tuple: 0}
    
    # Track path
    came_from = {}
    
    # Track visited
    closed_set = set()
    
    directions = get_directions()
    
    while open_set:
        f_score, g_score, current_tuple = heapq.heappop(open_set)
        
        # Goal reached
        if current_tuple == goal_tuple:
            # Reconstruct path
            path = []
            current = current_tuple
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return g_score, path
        
        # Skip if already visited
        if current_tuple in closed_set:
            continue
        
        closed_set.add(current_tuple)
        
        # Explore neighbors
        current_pos = {"x": current_tuple[0], "y": current_tuple[1]}
        for direction in directions:
            neighbor = get_next_position(current_pos, direction)
            neighbor_tuple = (neighbor["x"], neighbor["y"])
            
            # Out of bounds check
            if is_out_of_bounds(neighbor, board_width, board_height):
                continue
            
            # Obstacle check (allow goal position)
            if neighbor_tuple in obstacles:
                if not allow_goal_obstacle or neighbor_tuple != goal_tuple:
                    continue
            
            # Calculate cost
            base_cost = 1
            extra_cost = 0
            if cost_fn:
                extra_cost = cost_fn(current_pos, neighbor)
            
            tentative_g = g_score + base_cost + extra_cost
            
            # Skip if not better
            if neighbor_tuple in g_scores and tentative_g >= g_scores[neighbor_tuple]:
                continue
            
            # Best path so far
            g_scores[neighbor_tuple] = tentative_g
            came_from[neighbor_tuple] = current_tuple
            
            # Calculate f_score
            h_score = manhattan_distance(neighbor, goal)
            f_score = tentative_g + h_score
            
            heapq.heappush(open_set, (f_score, tentative_g, neighbor_tuple))
    
    return None, None


def flood_fill_count(start: typing.Dict, game_state: typing.Dict, obstacles: set, 
                     max_depth: int = None) -> int:
    """
    Count reachable spaces from start position using BFS.
    Optional max_depth limits how far to explore.
    """
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    
    visited = set()
    queue = deque([(start, 0)])
    visited.add((start["x"], start["y"]))
    
    directions = get_directions()
    
    while queue:
        current, depth = queue.popleft()
        
        # Depth limit reached
        if max_depth is not None and depth >= max_depth:
            continue
        
        for direction in directions:
            next_pos = get_next_position(current, direction)
            pos_tuple = (next_pos["x"], next_pos["y"])
            
            if pos_tuple in visited:
                continue
            if is_out_of_bounds(next_pos, board_width, board_height):
                continue
            if pos_tuple in obstacles:
                continue
            
            visited.add(pos_tuple)
            queue.append((next_pos, depth + 1))
    
    return len(visited)


def find_safe_path_to_tail(my_head: typing.Dict, my_body: typing.List, game_state: typing.Dict,
                           obstacles: set) -> typing.Optional[int]:
    """
    Find path to our own tail. This is useful for tail-chasing when we need to buy time.
    """
    if len(my_body) < 3:
        return None
    
    my_tail = my_body[-1]
    
    # Our tail will move, so the square it's on will become available
    # Create obstacles without our tail
    obstacles_without_tail = obstacles.copy()
    tail_tuple = (my_tail["x"], my_tail["y"])
    obstacles_without_tail.discard(tail_tuple)
    
    distance, _ = astar_pathfind(my_head, my_tail, game_state, obstacles_without_tail, 
                                 allow_goal_obstacle=True)
    
    return distance


# ============================================================================
# TERRITORY CONTROL (VORONOI)
# ============================================================================

def calculate_voronoi_territories(game_state: typing.Dict, obstacles: set) -> typing.Dict:
    """
    Calculate Voronoi territories - which cells each snake can reach first.
    Returns dict: {snake_id: territory_count}
    """
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    
    cell_owner = {}
    cell_distance = {}
    
    # Initialize BFS from all snake heads
    queue = deque()
    for snake in game_state["board"]["snakes"]:
        snake_id = snake.get("id")
        body = snake.get("body", [])
        if not snake_id or len(body) == 0:
            continue
        
        head = body[0]
        head_tuple = (head["x"], head["y"])
        queue.append((head_tuple, snake_id, 0))
        cell_owner[head_tuple] = snake_id
        cell_distance[head_tuple] = 0
    
    directions = get_directions()
    
    # BFS to fill board
    while queue:
        current_tuple, snake_id, distance = queue.popleft()
        current = {"x": current_tuple[0], "y": current_tuple[1]}
        
        for direction in directions:
            next_pos = get_next_position(current, direction)
            next_tuple = (next_pos["x"], next_pos["y"])
            
            if is_out_of_bounds(next_pos, board_width, board_height):
                continue
            
            if next_tuple in obstacles:
                continue
            
            new_distance = distance + 1
            
            # If unclaimed or we reach it at same time (contested)
            if next_tuple not in cell_distance or new_distance <= cell_distance[next_tuple]:
                if next_tuple not in cell_distance:
                    cell_owner[next_tuple] = snake_id
                    cell_distance[next_tuple] = new_distance
                    queue.append((next_tuple, snake_id, new_distance))
    
    # Count territories
    territory_count = {}
    for snake in game_state["board"]["snakes"]:
        snake_id = snake.get("id")
        if snake_id:
            territory_count[snake_id] = 0
    
    for pos, owner_id in cell_owner.items():
        territory_count[owner_id] = territory_count.get(owner_id, 0) + 1
    
    return territory_count


def calculate_future_voronoi(game_state: typing.Dict, my_id: str, my_next_pos: typing.Dict,
                             obstacles: set) -> typing.Dict:
    """
    Simulate Voronoi territories after we make a move.
    This helps predict territory gain/loss.
    """
    # Create simulated game state
    sim_state = {
        "board": {
            "width": game_state["board"]["width"],
            "height": game_state["board"]["height"],
            "snakes": []
        }
    }
    
    # Add our snake with new head position (simplified: just move head)
    for snake in game_state["board"]["snakes"]:
        snake_id = snake.get("id", "")
        body = snake.get("body", [])
        
        if snake_id == my_id and len(body) > 0:
            # Our new body: new head + old body minus tail
            new_body = [my_next_pos] + body[:-1]
            sim_state["board"]["snakes"].append({
                "id": snake_id,
                "body": new_body
            })
        else:
            # Other snakes unchanged (simplified)
            sim_state["board"]["snakes"].append(snake)
    
    # Calculate voronoi in simulated state
    sim_obstacles = get_all_snake_positions(sim_state, exclude_tails=True)
    return calculate_voronoi_territories(sim_state, sim_obstacles)


# ============================================================================
# OPPONENT MODELING
# ============================================================================

def get_opponent_info(game_state: typing.Dict, my_id: str) -> typing.List[typing.Dict]:
    """
    Get detailed information about all opponents.
    Returns list of dicts with: id, head, length, body, health
    """
    opponents = []
    
    for snake in game_state["board"]["snakes"]:
        snake_id = snake.get("id", "")
        body = snake.get("body", [])
        
        if snake_id != my_id and len(body) > 0:
            opponents.append({
                "id": snake_id,
                "head": body[0],
                "length": len(body),
                "body": body,
                "health": snake.get("health", 100)
            })
    
    return opponents


def predict_opponent_moves(opponent: typing.Dict, game_state: typing.Dict, 
                           obstacles: set, weighted: bool = False) -> typing.List[typing.Dict]:
    """
    Predict where opponent might move.
    If weighted=True, returns list of (position, probability) tuples.
    Otherwise returns list of possible positions.
    """
    head = opponent["head"]
    opp_id = opponent["id"]
    possible_moves = []
    
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    
    # Get opponent behavior profile
    behavior = opponent_history.get(opp_id, {"aggression": 0.5, "risk_tolerance": 0.5})
    
    for direction in get_directions():
        next_pos = get_next_position(head, direction)
        next_tuple = (next_pos["x"], next_pos["y"])
        
        # Basic safety check
        if is_out_of_bounds(next_pos, board_width, board_height):
            continue
        if next_tuple in obstacles:
            continue
        
        if weighted:
            # Calculate move probability based on behavior and space
            space = flood_fill_count(next_pos, game_state, obstacles, max_depth=10)
            probability = 0.25  # Base probability
            
            # Prefer moves with more space
            if space > opponent["length"] * 2:
                probability += 0.3
            elif space > opponent["length"]:
                probability += 0.1
            
            # Factor in aggression for center moves
            center_x, center_y = board_width // 2, board_height // 2
            if abs(next_pos["x"] - center_x) <= 2 and abs(next_pos["y"] - center_y) <= 2:
                probability += behavior["aggression"] * 0.2
            
            possible_moves.append((next_pos, probability))
        else:
            possible_moves.append(next_pos)
    
    # Normalize probabilities
    if weighted and len(possible_moves) > 0:
        total_prob = sum(p for _, p in possible_moves)
        possible_moves = [(pos, p/total_prob) for pos, p in possible_moves]
    
    return possible_moves


def is_head_to_head_risky(my_pos: typing.Dict, my_length: int, 
                          opponents: typing.List[typing.Dict],
                          game_state: typing.Dict, obstacles: set) -> bool:
    """
    Check if moving to my_pos risks a head-to-head collision with a longer/equal opponent.
    """
    for opponent in opponents:
        opp_head = opponent["head"]
        opp_length = opponent["length"]
        
        # Check if opponent can reach adjacent to my_pos
        possible_opp_moves = predict_opponent_moves(opponent, game_state, obstacles)
        
        for opp_next in possible_opp_moves:
            if manhattan_distance(my_pos, opp_next) <= 1:
                # They can collide with us
                if opp_length >= my_length:
                    return True
    
    return False


def calculate_opponent_cutoff_potential(my_pos: typing.Dict, opponent: typing.Dict,
                                       game_state: typing.Dict, obstacles: set) -> float:
    """
    Calculate how much we can reduce opponent's accessible space by moving to my_pos.
    Returns a score representing cut-off potential.
    """
    opp_head = opponent["head"]
    
    # Calculate opponent's current accessible space
    current_space = flood_fill_count(opp_head, game_state, obstacles, max_depth=15)
    
    # Simulate obstacles with my_pos added
    future_obstacles = obstacles.copy()
    future_obstacles.add((my_pos["x"], my_pos["y"]))
    
    # Calculate opponent's future accessible space
    future_space = flood_fill_count(opp_head, game_state, future_obstacles, max_depth=15)
    
    # Space reduction
    space_reduction = current_space - future_space
    
    # Bonus if we're blocking their path to food
    if len(game_state["board"]["food"]) > 0 and opponent["health"] < 50:
        for food in game_state["board"]["food"]:
            # Check if my_pos is on their path to food
            dist_opp_to_food = manhattan_distance(opp_head, food)
            dist_me_to_food = manhattan_distance(my_pos, food)
            dist_opp_to_me = manhattan_distance(opp_head, my_pos)
            
            # If we're between them and food
            if dist_opp_to_me + dist_me_to_food <= dist_opp_to_food + 2:
                space_reduction += 10  # Bonus for food blocking
    
    return space_reduction


# ============================================================================
# MULTI-TURN LOOKAHEAD
# ============================================================================

def minimax_evaluate(my_pos: typing.Dict, my_length: int, opponents: typing.List[typing.Dict],
                     game_state: typing.Dict, obstacles: set, depth: int = 2) -> float:
    """
    Multi-turn lookahead using minimax-style evaluation.
    Returns a score estimating position quality after 'depth' turns.
    """
    if depth == 0:
        # Base case: evaluate current position
        space = flood_fill_count(my_pos, game_state, obstacles, max_depth=None)
        return space
    
    # Simulate our next moves
    best_score = -float('inf')
    
    for direction in get_directions():
        next_pos = get_next_position(my_pos, direction)
        next_tuple = (next_pos["x"], next_pos["y"])
        
        board_width = game_state["board"]["width"]
        board_height = game_state["board"]["height"]
        
        # Skip invalid moves
        if is_out_of_bounds(next_pos, board_width, board_height):
            continue
        if next_tuple in obstacles:
            continue
        
        # Simulate opponent responses (worst case)
        worst_opponent_outcome = float('inf')
        
        for opp in opponents:
            opp_moves = predict_opponent_moves(opp, game_state, obstacles, weighted=False)
            if len(opp_moves) == 0:
                continue
            
            for opp_next in opp_moves:
                # Update obstacles with our move and opponent move
                sim_obstacles = obstacles.copy()
                sim_obstacles.add((next_pos["x"], next_pos["y"]))
                sim_obstacles.add((opp_next["x"], opp_next["y"]))
                
                # Recursive evaluation
                score = minimax_evaluate(next_pos, my_length, opponents, 
                                        game_state, sim_obstacles, depth - 1)
                worst_opponent_outcome = min(worst_opponent_outcome, score)
        
        if worst_opponent_outcome == float('inf'):
            worst_opponent_outcome = flood_fill_count(next_pos, game_state, obstacles, max_depth=None)
        
        best_score = max(best_score, worst_opponent_outcome)
    
    return best_score if best_score > -float('inf') else 0


def evaluate_move_sequence(start_pos: typing.Dict, directions: typing.List[str],
                          game_state: typing.Dict, obstacles: set) -> typing.Optional[float]:
    """
    Evaluate a sequence of moves to see if it leads to a good position.
    Returns the space available after the sequence, or None if invalid.
    """
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    
    current_pos = start_pos
    sim_obstacles = obstacles.copy()
    
    for direction in directions:
        next_pos = get_next_position(current_pos, direction)
        next_tuple = (next_pos["x"], next_pos["y"])
        
        if is_out_of_bounds(next_pos, board_width, board_height):
            return None
        if next_tuple in sim_obstacles:
            return None
        
        sim_obstacles.add((current_pos["x"], current_pos["y"]))
        current_pos = next_pos
    
    return flood_fill_count(current_pos, game_state, sim_obstacles, max_depth=None)


# ============================================================================
# TRAP DETECTION
# ============================================================================

def count_safe_exits(pos: typing.Dict, game_state: typing.Dict, obstacles: set) -> int:
    """Count how many safe adjacent squares exist from this position"""
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    
    safe_count = 0
    for direction in get_directions():
        next_pos = get_next_position(pos, direction)
        
        if is_out_of_bounds(next_pos, board_width, board_height):
            continue
        if (next_pos["x"], next_pos["y"]) in obstacles:
            continue
        
        safe_count += 1
    
    return safe_count


def is_corridor(pos: typing.Dict, game_state: typing.Dict, obstacles: set) -> bool:
    """
    Detect if position is in a corridor (only 2 exits, typically opposite directions).
    """
    exits = count_safe_exits(pos, game_state, obstacles)
    return exits <= 2


def estimate_trap_risk(pos: typing.Dict, game_state: typing.Dict, obstacles: set, 
                      my_length: int, opponents: typing.List[typing.Dict] = None) -> float:
    """
    Estimate how risky this position is (trap potential).
    Returns risk score (higher = more dangerous).
    """
    risk = 0.0
    
    # Check exits
    exits = count_safe_exits(pos, game_state, obstacles)
    if exits == 0:
        risk += 1000  # Dead end
    elif exits == 1:
        risk += 500   # Extreme trap
    elif exits == 2:
        risk += 100   # Corridor
        
        # Check if corridor is a dead-end corridor (dangerous)
        # Test 2-3 moves ahead in corridor
        test_dirs = []
        for direction in get_directions():
            test_pos = get_next_position(pos, direction)
            test_tuple = (test_pos["x"], test_pos["y"])
            if not is_out_of_bounds(test_pos, game_state["board"]["width"], 
                                   game_state["board"]["height"]):
                if test_tuple not in obstacles:
                    test_dirs.append(direction)
        
        # Simulate moving through corridor
        if len(test_dirs) >= 2:
            for direction in test_dirs:
                future_space = evaluate_move_sequence(pos, [direction] * 3, game_state, obstacles)
                if future_space is not None and future_space < my_length:
                    risk += 200  # Corridor leads to trap
    
    # Check accessible space
    accessible = flood_fill_count(pos, game_state, obstacles, max_depth=my_length + 3)
    if accessible < my_length:
        risk += 500  # Not enough space to survive
    elif accessible < my_length * 1.5:
        risk += 200  # Tight squeeze
    
    # Check wall proximity
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    
    walls_adjacent = 0
    if pos["x"] == 0 or pos["x"] == board_width - 1:
        walls_adjacent += 1
    if pos["y"] == 0 or pos["y"] == board_height - 1:
        walls_adjacent += 1
    
    if walls_adjacent >= 2:
        risk += 150  # Corner
    elif walls_adjacent == 1:
        risk += 30   # Wall-hugging
    
    # Check if opponents can trap us here
    if opponents:
        for opp in opponents:
            if opp["length"] >= my_length:
                opp_dist = manhattan_distance(pos, opp["head"])
                if opp_dist <= 3 and exits <= 2:
                    risk += 100  # Opponent nearby and we have few exits
    
    return risk


# ============================================================================
# GAME PHASE DETERMINATION
# ============================================================================

def determine_game_phase(game_state: typing.Dict) -> str:
    """
    Determine current game phase based on number of alive snakes and board state.
    Phases: 'early', 'mid', 'late', '1v1'
    """
    alive_snakes = len(game_state["board"]["snakes"])
    turn = game_state.get("turn", 0)
    
    if alive_snakes == 2:
        return "1v1"
    elif alive_snakes <= 3:
        return "late"
    elif turn < 40:
        return "early"
    else:
        return "mid"


# ============================================================================
# FOOD STRATEGY
# ============================================================================

def evaluate_food_urgency(my_health: int, my_length: int, opponents: typing.List[typing.Dict],
                         game_phase: str) -> float:
    """
    Determine how urgently we need food.
    Returns urgency score (0 = not urgent, 1 = very urgent).
    """
    urgency = 0.0
    
    # Health-based urgency
    if my_health < 20:
        urgency += 1.0
    elif my_health < 40:
        urgency += 0.7
    elif my_health < 60:
        urgency += 0.4
    elif my_health < 80:
        urgency += 0.2
    
    # Length-based urgency (want to be longest)
    if len(opponents) > 0:
        max_opp_length = max(opp["length"] for opp in opponents)
        if my_length < max_opp_length:
            urgency += 0.5
        elif my_length < max_opp_length + 2:
            urgency += 0.3
    
    # Phase-based urgency
    if game_phase == "early":
        urgency += 0.4  # Prioritize growth
    elif game_phase == "1v1" and my_health > 70:
        urgency -= 0.5  # Avoid food in 1v1 if healthy
    
    return min(urgency, 1.0)


def find_best_food(my_head: typing.Dict, food_list: typing.List[typing.Dict],
                  game_state: typing.Dict, obstacles: set,
                  opponents: typing.List[typing.Dict], my_length: int) -> typing.Optional[typing.Tuple]:
    """
    Find the best food to pursue with advanced strategic analysis.
    Returns (food_pos, distance) or None if no reachable food.
    """
    if len(food_list) == 0:
        return None
    
    food_scores = []
    
    for food in food_list:
        # Calculate our distance to food
        my_dist, my_path = astar_pathfind(my_head, food, game_state, obstacles, allow_goal_obstacle=True)
        
        if my_dist is None:
            continue  # Unreachable
        
        # Calculate closest opponent distance to this food
        min_opp_dist = float('inf')
        closest_opp = None
        for opp in opponents:
            opp_dist, _ = astar_pathfind(opp["head"], food, game_state, obstacles, allow_goal_obstacle=True)
            if opp_dist is not None and opp_dist < min_opp_dist:
                min_opp_dist = opp_dist
                closest_opp = opp
        
        # Score: prefer closer food and food we can reach before opponents
        score = -my_dist
        
        # Competition analysis
        if my_dist < min_opp_dist:
            score += 50  # We'll reach it first
            # Extra bonus if opponent is longer and needs food (deny them)
            if closest_opp and closest_opp["length"] > my_length:
                score += 30
        elif my_dist == min_opp_dist:
            # Contested - only go if we're longer
            if closest_opp and my_length > closest_opp["length"]:
                score += 20  # We win head-to-head
            else:
                score -= 40  # Risky, avoid
        else:
            score -= 30  # They'll reach it first
            # Don't chase food they'll get
            if min_opp_dist < my_dist - 2:
                score -= 50
        
        # Check if path to food goes through dangerous areas
        if my_path:
            danger_on_path = 0
            for path_pos in my_path[:min(5, len(my_path))]:
                path_dict = {"x": path_pos[0], "y": path_pos[1]}
                exits = count_safe_exits(path_dict, game_state, obstacles)
                if exits <= 2:
                    danger_on_path += 1
            
            score -= danger_on_path * 15  # Penalize dangerous paths
        
        # Check space after eating food
        food_obstacles = obstacles.copy()
        food_obstacles.add((food["x"], food["y"]))  # Temporarily treat food as obstacle
        space_after = flood_fill_count(food, game_state, food_obstacles, max_depth=15)
        if space_after < my_length:
            score -= 100  # Food is in a trap!
        
        food_scores.append((food, my_dist, score))
    
    if len(food_scores) == 0:
        return None
    
    # Return best food
    best_food = max(food_scores, key=lambda x: x[2])
    return (best_food[0], best_food[1])


# ============================================================================
# AGGRESSIVE TACTICS
# ============================================================================

def evaluate_hunting_opportunity(my_pos: typing.Dict, my_length: int,
                                opponent: typing.Dict, game_state: typing.Dict,
                                obstacles: set) -> float:
    """
    Evaluate if we should hunt/chase this opponent.
    Returns hunting score (higher = better opportunity).
    """
    score = 0.0
    
    opp_head = opponent["head"]
    opp_length = opponent["length"]
    
    # Only hunt if we're longer
    if my_length <= opp_length:
        return 0.0
    
    length_advantage = my_length - opp_length
    score += length_advantage * 10
    
    # Distance to opponent
    distance = manhattan_distance(my_pos, opp_head)
    if distance < 3:
        score += 50
    elif distance < 5:
        score += 30
    elif distance < 8:
        score += 10
    
    # Check if opponent is trapped or low on space
    opp_space = flood_fill_count(opp_head, game_state, obstacles, max_depth=15)
    if opp_space < opp_length:
        score += 100  # They're already trapped
    elif opp_space < opp_length * 1.5:
        score += 50   # Tight spot
    
    # Check opponent health
    opp_health = opponent.get("health", 100)
    if opp_health < 30:
        score += 40  # They're desperate for food
    
    return score


def find_cutoff_moves(my_pos: typing.Dict, opponent: typing.Dict,
                     game_state: typing.Dict, obstacles: set) -> float:
    """
    Calculate how effective this move is at cutting off opponent's space.
    Returns cutoff score.
    """
    cutoff_score = calculate_opponent_cutoff_potential(my_pos, opponent, game_state, obstacles)
    
    return cutoff_score


# ============================================================================
# MOVE SCORING SYSTEM
# ============================================================================

def score_move(direction: str, my_head: typing.Dict, my_body: typing.List, my_length: int,
              my_health: int, my_id: str, game_state: typing.Dict, obstacles: set,
              opponents: typing.List[typing.Dict], game_phase: str,
              voronoi_territories: typing.Dict) -> float:
    """
    Comprehensive move scoring function.
    Returns a score for the move (higher = better).
    """
    next_pos = get_next_position(my_head, direction)
    next_tuple = (next_pos["x"], next_pos["y"])
    
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    
    score = 0.0
    
    # -------------------------------------------------------------------------
    # IMMEDIATE SAFETY CHECKS (hard disqualifications)
    # -------------------------------------------------------------------------
    
    # Out of bounds
    if is_out_of_bounds(next_pos, board_width, board_height):
        return -100000
    
    # Reverse into neck (CHECK THIS FIRST before general body collision)
    if len(my_body) > 1:
        neck = my_body[1]
        if next_pos["x"] == neck["x"] and next_pos["y"] == neck["y"]:
            return -100000
    
    # Snake body collision
    if next_tuple in obstacles:
        return -100000
    
    # Head-to-head collision risk
    if is_head_to_head_risky(next_pos, my_length, opponents, game_state, obstacles):
        return -100000
    
    # -------------------------------------------------------------------------
    # SPACE EVALUATION
    # -------------------------------------------------------------------------
    
    # Flood fill to count accessible space
    future_obstacles = obstacles.copy()
    reachable_space = flood_fill_count(next_pos, game_state, future_obstacles, max_depth=None)
    
    # Must have minimum space to survive
    if reachable_space < my_length:
        score -= 5000
    else:
        # Reward proportional to space (important in all phases)
        if game_phase == "early":
            score += reachable_space * 3
        elif game_phase == "mid":
            score += reachable_space * 5
        else:  # late or 1v1
            score += reachable_space * 8
    
    # -------------------------------------------------------------------------
    # TRAP RISK EVALUATION
    # -------------------------------------------------------------------------
    
    trap_risk = estimate_trap_risk(next_pos, game_state, future_obstacles, my_length, opponents)
    score -= trap_risk
    
    # -------------------------------------------------------------------------
    # MULTI-TURN LOOKAHEAD (expensive, only in critical situations)
    # -------------------------------------------------------------------------
    
    # Use lookahead in late game or when space is tight
    if game_phase in ["late", "1v1"] or reachable_space < my_length * 3:
        # Lightweight lookahead (depth 2)
        lookahead_score = minimax_evaluate(next_pos, my_length, opponents[:2],  # Limit to 2 opponents
                                          game_state, future_obstacles, depth=1)
        score += lookahead_score * 0.5  # Modest weight to avoid over-optimization
    
    # -------------------------------------------------------------------------
    # TERRITORY CONTROL (VORONOI)
    # -------------------------------------------------------------------------
    
    my_territory = voronoi_territories.get(my_id, 0)
    future_voronoi = calculate_future_voronoi(game_state, my_id, next_pos, obstacles)
    future_my_territory = future_voronoi.get(my_id, 0)
    
    territory_change = future_my_territory - my_territory
    
    # Calculate opponent territory loss
    opp_territory_loss = 0
    for opp in opponents:
        opp_id = opp["id"]
        old_territory = voronoi_territories.get(opp_id, 0)
        new_territory = future_voronoi.get(opp_id, 0)
        opp_territory_loss += (old_territory - new_territory)
    
    # Weight by game phase
    if game_phase == "1v1":
        score += territory_change * 20
        score += opp_territory_loss * 15
    elif game_phase == "late":
        score += territory_change * 15
        score += opp_territory_loss * 10
    elif game_phase == "mid":
        score += territory_change * 8
        score += opp_territory_loss * 5
    else:  # early
        score += territory_change * 3
        score += opp_territory_loss * 2
    
    # -------------------------------------------------------------------------
    # FOOD SEEKING
    # -------------------------------------------------------------------------
    
    food_list = game_state["board"]["food"]
    food_urgency = evaluate_food_urgency(my_health, my_length, opponents, game_phase)
    
    if food_urgency > 0.1 and len(food_list) > 0:
        best_food = find_best_food(next_pos, food_list, game_state, future_obstacles, opponents, my_length)
        
        if best_food is not None:
            food_pos, food_dist = best_food
            
            # Scale by urgency
            food_weight = food_urgency * 100
            
            # Closer food is better
            score -= food_dist * food_weight
            
            # Bonus for very close food
            if food_dist <= 2:
                score += 200 * food_urgency
    else:
        # Avoid food if not urgent (especially in late game)
        if len(food_list) > 0:
            closest_food_dist = min(manhattan_distance(next_pos, f) for f in food_list)
            if game_phase in ["late", "1v1"] and my_health > 70:
                score += closest_food_dist * 2  # Stay away from food
    
    # -------------------------------------------------------------------------
    # TAIL CHASING (ESCAPE MECHANISM)
    # -------------------------------------------------------------------------
    
    # If low on space and health is OK, consider chasing our tail
    if reachable_space < my_length * 2 and my_health > 40:
        tail_dist = find_safe_path_to_tail(next_pos, my_body, game_state, future_obstacles)
        if tail_dist is not None:
            score += 100 / (tail_dist + 1)
    
    # -------------------------------------------------------------------------
    # AGGRESSIVE TACTICS
    # -------------------------------------------------------------------------
    
    for opp in opponents:
        opp_id = opp["id"]
        opp_length = opp["length"]
        
        # Hunting (when we're longer)
        if my_length > opp_length:
            hunting_score = evaluate_hunting_opportunity(next_pos, my_length, opp, 
                                                        game_state, future_obstacles)
            
            # Weight by game phase
            if game_phase == "1v1":
                score += hunting_score * 2.0
            elif game_phase == "late":
                score += hunting_score * 1.5
            elif game_phase == "mid":
                score += hunting_score * 0.8
            else:  # early
                score += hunting_score * 0.3
        
        # Cutoff moves (reduce opponent space)
        cutoff_potential = find_cutoff_moves(next_pos, opp, game_state, obstacles)
        
        if cutoff_potential > 5:
            if game_phase in ["late", "1v1"]:
                score += cutoff_potential * 10
            elif game_phase == "mid":
                score += cutoff_potential * 5
            else:  # early
                score += cutoff_potential * 2
        
        # Maintain distance if they're longer (defensive)
        if my_length < opp_length:
            opp_dist = manhattan_distance(next_pos, opp["head"])
            if opp_dist < 4:
                score += opp_dist * 30  # Stay away
    
    # -------------------------------------------------------------------------
    # CENTER CONTROL (early game)
    # -------------------------------------------------------------------------
    
    if game_phase == "early" and my_health > 60:
        center_x = board_width // 2
        center_y = board_height // 2
        center_dist = manhattan_distance(next_pos, {"x": center_x, "y": center_y})
        score -= center_dist * 2
    
    # -------------------------------------------------------------------------
    # AVOID WALLS (unless necessary)
    # -------------------------------------------------------------------------
    
    wall_penalty = 0
    if next_pos["x"] == 0 or next_pos["x"] == board_width - 1:
        wall_penalty += 10
    if next_pos["y"] == 0 or next_pos["y"] == board_height - 1:
        wall_penalty += 10
    
    score -= wall_penalty
    
    # -------------------------------------------------------------------------
    # BONUS FOR MULTIPLE ESCAPE ROUTES
    # -------------------------------------------------------------------------
    
    safe_exits = count_safe_exits(next_pos, game_state, future_obstacles)
    if safe_exits >= 3:
        score += 80  # Good position with many options
    elif safe_exits == 2:
        score += 20
    
    return score


# ============================================================================
# MAIN MOVE FUNCTION
# ============================================================================

def update_opponent_behavior(game_state: typing.Dict):
    """
    Track opponent behavior patterns across turns for better prediction.
    """
    for snake in game_state["board"]["snakes"]:
        snake_id = snake.get("id")
        if not snake_id or snake_id == game_state["you"].get("id"):
            continue
        
        body = snake.get("body", [])
        if len(body) < 2:
            continue
        
        head = body[0]
        neck = body[1]
        
        # Determine move direction
        dx = head["x"] - neck["x"]
        dy = head["y"] - neck["y"]
        
        direction = None
        if dx == 1:
            direction = "right"
        elif dx == -1:
            direction = "left"
        elif dy == 1:
            direction = "up"
        elif dy == -1:
            direction = "down"
        
        if direction:
            opponent_history[snake_id]["moves"].append(direction)
            # Keep only recent moves
            if len(opponent_history[snake_id]["moves"]) > 10:
                opponent_history[snake_id]["moves"].pop(0)


def move(game_state: typing.Dict) -> typing.Dict:
    """
    Main move decision function.
    Evaluates all possible moves and selects the best one based on comprehensive scoring.
    """
    # Update opponent behavior tracking
    update_opponent_behavior(game_state)
    
    # Extract game state information
    my_id = game_state["you"].get("id", "")
    my_body = game_state["you"]["body"]
    my_head = my_body[0]
    my_length = len(my_body)
    my_health = game_state["you"]["health"]
    
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    
    # Get game phase
    game_phase = determine_game_phase(game_state)
    
    # Get obstacles (snake bodies)
    obstacles = get_all_snake_positions(game_state, exclude_tails=True)
    
    # Get opponent information
    opponents = get_opponent_info(game_state, my_id)
    
    # Calculate Voronoi territories
    voronoi_territories = calculate_voronoi_territories(game_state, obstacles)
    
    # Score all possible moves
    directions = get_directions()
    move_scores = {}
    
    for direction in directions:
        move_scores[direction] = score_move(
            direction, my_head, my_body, my_length, my_health, my_id,
            game_state, obstacles, opponents, game_phase, voronoi_territories
        )
    
    # Select best move
    valid_moves = [d for d in directions if move_scores[d] > -50000]
    
    if len(valid_moves) == 0:
        # Desperate situation - pick least bad move
        best_move = max(directions, key=lambda d: move_scores[d])
    else:
        best_move = max(valid_moves, key=lambda d: move_scores[d])
    
    # Final safety check
    final_pos = get_next_position(my_head, best_move)
    if is_out_of_bounds(final_pos, board_width, board_height):
        # Emergency: try to find any in-bounds move
        for emergency_dir in directions:
            emergency_pos = get_next_position(my_head, emergency_dir)
            if not is_out_of_bounds(emergency_pos, board_width, board_height):
                best_move = emergency_dir
                break
    
    return {"move": best_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server
    
    run_server({"info": info, "start": start, "move": move, "end": end})
