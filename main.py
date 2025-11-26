import typing
from collections import deque
import heapq

# --------------------------
# Helpers
# --------------------------

def neighbors(x, y):
    return [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def flood_fill(game_state, start, blocked):
    """Count reachable tiles starting from 'start' using BFS."""
    width = game_state["board"]["width"]
    height = game_state["board"]["height"]

    q = deque([start])
    visited = {start}

    while q:
        x, y = q.popleft()
        for nx, ny in neighbors(x, y):
            # IN-BOUNDS CHECK
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            # TILE NOT SAFE
            if (nx, ny) in blocked:
                continue
            if (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            q.append((nx, ny))

    return len(visited)

def build_blocked_tiles(game_state):
    """All snake body tiles (including your own), excluding tails that will move."""
    blocked = set()
    for snake in game_state["board"]["snakes"]:
        body = snake["body"]
        for i, p in enumerate(body):
            # Skip tail unless snake just ate (duplicate tail position means it ate)
            if i == len(body) - 1 and len(body) > 1:
                # Check if tail is duplicate (snake just ate)
                tail_pos = (body[-1]["x"], body[-1]["y"])
                second_to_last = (body[-2]["x"], body[-2]["y"])
                if tail_pos != second_to_last:
                    continue  # Skip tail, it will move
            blocked.add((p["x"], p["y"]))
    return blocked

def get_opponent_head_zones(game_state, my_head):
    """Get tiles adjacent to opponent snake heads where head-to-head collisions could happen."""
    my_length = len(game_state["you"]["body"])
    danger_zones = set()
    
    for snake in game_state["board"]["snakes"]:
        # Skip ourselves
        if snake["id"] == game_state["you"]["id"]:
            continue
        
        # Avoid head-to-head with snakes that are NOT smaller (equal or larger)
        # If they're equal or larger, we lose or it's random - both bad
        if len(snake["body"]) >= my_length:
            head = (snake["head"]["x"], snake["head"]["y"])
            # Add all possible moves for this opponent
            for nx, ny in neighbors(head[0], head[1]):
                danger_zones.add((nx, ny))
    
    return danger_zones

def get_hazards(game_state):
    """Get all hazard positions on the board."""
    hazards = set()
    if "hazards" in game_state["board"]:
        for h in game_state["board"]["hazards"]:
            hazards.add((h["x"], h["y"]))
    return hazards

def analyze_opponents(game_state):
    """Analyze all opponents and identify vulnerable ones."""
    my_id = game_state["you"]["id"]
    my_head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    my_length = len(game_state["you"]["body"])
    
    opponents = []
    for snake in game_state["board"]["snakes"]:
        if snake["id"] == my_id:
            continue
        
        opponents.append({
            "id": snake["id"],
            "head": (snake["head"]["x"], snake["head"]["y"]),
            "health": snake["health"],
            "length": len(snake["body"]),
            "is_desperate": snake["health"] < 30,
            "is_hungry": snake["health"] < 60,
            "distance_from_me": manhattan_distance(
                my_head, 
                (snake["head"]["x"], snake["head"]["y"])
            )
        })
    
    return opponents

def find_their_closest_food(opponent_head, food_list):
    """Find which food the opponent is likely targeting."""
    if not food_list:
        return None, float('inf')
    
    closest_to_them = None
    min_dist_to_them = float('inf')
    
    for food in food_list:
        food_pos = (food["x"], food["y"])
        dist_to_them = manhattan_distance(opponent_head, food_pos)
        
        if dist_to_them < min_dist_to_them:
            min_dist_to_them = dist_to_them
            closest_to_them = food_pos
    
    return closest_to_them, min_dist_to_them

def should_block_opponent_food(game_state, opponent, food_pos):
    """Decide if we should block an opponent's path to food."""
    my_head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    my_length = len(game_state["you"]["body"])
    my_health = game_state["you"]["health"]
    
    my_dist_to_food = manhattan_distance(my_head, food_pos)
    their_dist_to_food = manhattan_distance(opponent["head"], food_pos)
    
    # Only block if:
    # 1. They're desperate for food (health < 30)
    # 2. We're longer (we win head-to-head)
    # 3. We can reach food first or at same time
    # 4. We don't desperately need food ourselves
    
    if opponent["health"] < 30 and \
       my_length > opponent["length"] and \
       my_dist_to_food <= their_dist_to_food + 1 and \
       my_health > 40:
        return True
    
    return False

def determine_game_phase(game_state):
    """
    Determine what phase of the game we're in.
    Returns: 'early', 'mid', or 'late'
    """
    turn = game_state["turn"]
    num_snakes_alive = len(game_state["board"]["snakes"])
    board_size = game_state["board"]["width"] * game_state["board"]["height"]
    
    # Method 1: Turn-based (baseline)
    if turn < 50:
        phase = "early"
    elif turn < 200:
        phase = "mid"
    else:
        phase = "late"
    
    # Method 2: Snake count (better indicator)
    if num_snakes_alive >= 4:
        phase = "early"  # Still crowded
    elif num_snakes_alive == 2:
        phase = "late"   # 1v1 situation
    
    # Method 3: Board density
    total_snake_length = sum(len(snake["body"]) for snake in game_state["board"]["snakes"])
    board_density = total_snake_length / board_size
    
    if board_density > 0.5:
        phase = "late"  # Board is cramped, survival mode
    
    return phase

def determine_snake_mode(game_state, phase, opponents):
    """
    Determine whether to play aggressive or defensive.
    Returns: 'aggressive', 'defensive', or 'balanced'
    """
    my_health = game_state["you"]["health"]
    my_length = len(game_state["you"]["body"])
    
    if not opponents:
        return "defensive"  # Solo snake, just survive
    
    # Calculate relative strength
    avg_opponent_length = sum(opp["length"] for opp in opponents) / len(opponents)
    longest_opponent = max(opp["length"] for opp in opponents)
    
    am_i_biggest = my_length > longest_opponent
    am_i_strong = my_length > avg_opponent_length
    
    # DECISION TREE
    
    # If my health is critical, always defensive
    if my_health < 25:
        return "defensive"
    
    # Early game
    if phase == "early":
        if my_health < 60:
            return "defensive"  # Need food to grow
        else:
            return "balanced"   # Opportunistic
    
    # Mid game
    elif phase == "mid":
        if am_i_biggest and my_health > 50:
            return "aggressive"  # Capitalize on advantage
        elif am_i_strong:
            return "balanced"    # Selectively aggressive
        else:
            return "defensive"   # Catch up mode
    
    # Late game
    else:
        if am_i_biggest:
            return "aggressive"  # Hunt down remaining snakes
        elif len(opponents) == 1:  # 1v1
            if my_length > opponents[0]["length"]:
                return "aggressive"  # Go for the win
            else:
                return "defensive"   # Avoid until stronger
        else:
            return "balanced"

def find_closest_food(head, food_list):
    """Find the closest food using Manhattan distance."""
    if not food_list:
        return None
    
    closest = None
    min_dist = float('inf')
    
    for food in food_list:
        dist = manhattan_distance(head, (food["x"], food["y"]))
        if dist < min_dist:
            min_dist = dist
            closest = (food["x"], food["y"])
    
    return closest

def evaluate_food_competitively(head, food_list, game_state, my_length):
    """
    Competitive food selection - prioritizes food you can reach before enemies.
    
    How it works:
    1. For each food, calculate YOUR distance (using A*)
    2. For each food, calculate ENEMY distances (Manhattan - faster estimate)
    3. Score each food based on competitive advantage
    4. Return best food to target
    
    Scoring factors:
    - Can you reach it first? (BIG bonus)
    - Is it closer to you than enemies? (bonus)
    - Would taking it block/deny enemy access? (bonus)
    - Is it central/strategic? (small bonus)
    """
    if not food_list:
        return None
    
    my_head = head
    enemies = [s for s in game_state["board"]["snakes"] 
               if s["id"] != game_state["you"]["id"]]
    
    best_food = None
    best_score = -float('inf')
    
    print(f"  Evaluating {len(food_list)} food options competitively...")
    
    for food in food_list:
        food_pos = (food["x"], food["y"])
        
        # YOUR distance to this food (Manhattan as quick estimate)
        my_dist = manhattan_distance(my_head, food_pos)
        
        # Find closest ENEMY distance to this food
        min_enemy_dist = float('inf')
        closest_enemy_length = 0
        
        for enemy in enemies:
            enemy_head = (enemy["head"]["x"], enemy["head"]["y"])
            enemy_dist = manhattan_distance(enemy_head, food_pos)
            
            if enemy_dist < min_enemy_dist:
                min_enemy_dist = enemy_dist
                closest_enemy_length = len(enemy["body"])
        
        # SCORING SYSTEM
        score = 0
        
        # 1. BASE: Prefer closer food
        score += (20 - my_dist)
        
        # 2. COMPETITIVE ADVANTAGE: Can you reach it first?
        if min_enemy_dist == float('inf'):
            # No enemies - any food is good
            score += 50
        elif my_dist < min_enemy_dist:
            # YOU'RE CLOSER - big bonus!
            advantage = min_enemy_dist - my_dist
            score += 100 + (advantage * 10)
            print(f"    Food at {food_pos}: YOU win race by {advantage} steps! (+{100 + advantage * 10})")
        elif my_dist == min_enemy_dist:
            # TIE - depends on size
            if my_length > closest_enemy_length:
                score += 50  # You're bigger, might win head-to-head
                print(f"    Food at {food_pos}: Tied distance but you're bigger (+50)")
            else:
                score -= 50  # They're bigger or equal, risky
                print(f"    Food at {food_pos}: Tied but enemy is bigger (-50)")
        else:
            # ENEMY IS CLOSER - penalty but not impossible
            disadvantage = my_dist - min_enemy_dist
            score -= (disadvantage * 20)
            print(f"    Food at {food_pos}: Enemy closer by {disadvantage} (-{disadvantage * 20})")
        
        # 3. DENIAL VALUE: If you take it, how much does it hurt enemies?
        # Count how many enemies have this as their closest food
        enemies_targeting = 0
        for enemy in enemies:
            enemy_head = (enemy["head"]["x"], enemy["head"]["y"])
            # Check if this food is closer to enemy than you are
            if manhattan_distance(enemy_head, food_pos) < manhattan_distance(enemy_head, my_head):
                enemies_targeting += 1
        
        if enemies_targeting > 0:
            score += enemies_targeting * 15
            print(f"    Food at {food_pos}: Denies {enemies_targeting} enemies (+{enemies_targeting * 15})")
        
        # 4. STRATEGIC POSITION: Central food is more valuable
        board_width = game_state["board"]["width"]
        board_height = game_state["board"]["height"]
        center_x, center_y = board_width // 2, board_height // 2
        centrality = 10 - manhattan_distance(food_pos, (center_x, center_y))
        score += centrality * 2
        
        print(f"    Food at {food_pos}: Total score = {score:.1f}")
        
        if score > best_score:
            best_score = score
            best_food = food_pos
    
    if best_food:
        print(f"  → BEST COMPETITIVE FOOD: {best_food} (score: {best_score:.1f})")
    
    return best_food

def a_star_pathfind(start, goal, game_state, blocked, hazards):
    """
    A* pathfinding algorithm to find the shortest path from start to goal.
    
    Returns:
        - List of positions representing the path (including start and goal)
        - Empty list if no path exists
    
    How A* helps:
        1. Finds OPTIMAL path to food (shortest + safest)
        2. Accounts for obstacles dynamically
        3. More efficient than random movement
        4. Helps avoid getting trapped
    """
    width = game_state["board"]["width"]
    height = game_state["board"]["height"]
    
    # Priority queue: (f_score, counter, position, path)
    # f_score = g_score + h_score (actual cost + heuristic)
    counter = 0
    heap = [(0, counter, start, [start])]
    visited = {start}
    
    # g_score: actual cost from start to current position
    g_scores = {start: 0}
    
    while heap:
        f_score, _, current, path = heapq.heappop(heap)
        
        # Reached goal
        if current == goal:
            return path
        
        # Explore neighbors
        for nx, ny in neighbors(current[0], current[1]):
            neighbor = (nx, ny)
            
            # Out of bounds
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            
            # Blocked by snake body
            if neighbor in blocked:
                continue
            
            # Already visited with better score
            if neighbor in visited:
                continue
            
            # Calculate cost
            # Base cost is 1, but add penalty for hazards
            move_cost = 1
            if neighbor in hazards:
                move_cost += 5  # Avoid hazards unless necessary
            
            g_score = g_scores[current] + move_cost
            h_score = manhattan_distance(neighbor, goal)
            f_score = g_score + h_score
            
            # Add to exploration queue
            visited.add(neighbor)
            g_scores[neighbor] = g_score
            counter += 1
            heapq.heappush(heap, (f_score, counter, neighbor, path + [neighbor]))
    
    # No path found
    return []

def get_direction_from_path(current_pos, path):
    """Convert A* path to a move direction."""
    if len(path) < 2:
        return None
    
    next_pos = path[1]  # path[0] is current position
    
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]
    
    if dx == 1:
        return "right"
    elif dx == -1:
        return "left"
    elif dy == 1:
        return "up"
    elif dy == -1:
        return "down"
    
    return None

# --------------------------
# Main survival logic
# --------------------------

def choose_move(game_state):

    head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    width = game_state["board"]["width"]
    height = game_state["board"]["height"]
    my_health = game_state["you"]["health"]
    my_length = len(game_state["you"]["body"])

    # Analyze game state
    opponents = analyze_opponents(game_state)
    phase = determine_game_phase(game_state)
    mode = determine_snake_mode(game_state, phase, opponents)
    
    # GROWTH THRESHOLD: Stop aggressive growth to avoid self-trapping
    # Calculate based on board size - once we reach 40% of board, be more cautious
    board_size = width * height
    growth_threshold = int(board_size * 0.40)  # Stop aggressive growth at 40% of board
    max_safe_length = int(board_size * 0.60)   # Never exceed 60% of board
    
    is_growth_phase = my_length < growth_threshold
    is_dangerous_length = my_length > max_safe_length
    
    print(f"\nTurn {game_state['turn']}: Phase={phase.upper()}, Mode={mode.upper()}")
    print(f"  My stats: Length={my_length}/{growth_threshold} (growth threshold), Health={my_health}")
    print(f"  Growth phase: {is_growth_phase}, Dangerous length: {is_dangerous_length}")
    if opponents:
        print(f"  Opponents: {len(opponents)} alive, Lengths={[o['length'] for o in opponents]}, Healths={[o['health'] for o in opponents]}")

    blocked = build_blocked_tiles(game_state)
    danger_zones = get_opponent_head_zones(game_state, head)
    hazards = get_hazards(game_state)
    food_list = game_state["board"]["food"]
    
    # COMPETITIVE FOOD SELECTION: Choose food strategically
    # Use competitive evaluation if there are enemies, otherwise just get closest
    num_enemies = len(opponents)
    
    # GROWTH-FOCUSED STRATEGY
    # In growth phase: aggressively seek food and block opponents
    # After threshold: only eat when necessary (health < 70)
    
    if is_growth_phase:
        print("  🌱 GROWTH MODE: Aggressively seeking food")
        food_threshold = 90  # Almost always seek food
        space_requirement_multiplier = 0.8  # Accept tighter spaces
        # Use competitive food selection in growth mode
        if num_enemies > 0:
            closest_food = evaluate_food_competitively(head, food_list, game_state, my_length)
        else:
            closest_food = find_closest_food(head, food_list)
        seek_food_aggressively = True
        target_food = closest_food  # Already set above with competitive evaluation
    
    elif is_dangerous_length:
        print("  ⚠️ DANGEROUS LENGTH: Avoiding food, need more space")
        food_threshold = 30  # Only eat if desperate
        space_requirement_multiplier = 2.0  # Need lots of space
        seek_food_aggressively = False
        target_food = None  # Don't seek food unless critical health
        
        if my_health < 30:
            if num_enemies > 0:
                target_food = evaluate_food_competitively(head, food_list, game_state, my_length)
            else:
                target_food = find_closest_food(head, food_list)
    
    else:
        # Post-growth phase: maintain size, use mode-specific behavior
        print(f"  🎯 MAINTENANCE MODE: Using {mode} strategy")
        if mode == "aggressive":
            food_threshold = 40
            space_requirement_multiplier = 0.7
            seek_food_aggressively = False
        elif mode == "defensive":
            food_threshold = 80
            space_requirement_multiplier = 1.5
            seek_food_aggressively = False
        else:  # balanced
            food_threshold = 60
            space_requirement_multiplier = 1.0
            seek_food_aggressively = False
        
        # Use competitive food selection
        if num_enemies > 0 and my_health < 80:
            target_food = evaluate_food_competitively(head, food_list, game_state, my_length)
        else:
            target_food = find_closest_food(head, food_list)
    
    closest_food = target_food

    # ALL POSSIBLE MOVES
    moves = {
        "up":    (head[0], head[1] + 1),
        "down":  (head[0], head[1] - 1),
        "left":  (head[0] - 1, head[1]),
        "right": (head[0] + 1, head[1]),
    }

    safe_moves = {}

    # STEP 1: WALL, BODY, & HAZARD CHECK
    for move, (nx, ny) in moves.items():
        # WALL AVOIDANCE
        if nx < 0 or ny < 0 or nx >= width or ny >= height:
            print(f"  {move} -> WALL (would be at {nx},{ny})")
            continue

        # BODY COLLISION
        if (nx, ny) in blocked:
            print(f"  {move} -> BLOCKED by body at {nx},{ny}")
            continue
        
        # HAZARD AVOIDANCE (unless desperate - health check below)
        if (nx, ny) in hazards and my_health > 20:
            print(f"  {move} -> HAZARD at {nx},{ny} (avoiding unless desperate)")
            continue

        safe_moves[move] = (nx, ny)

    print(f"Turn {game_state['turn']}: Head at {head}, Safe moves: {list(safe_moves.keys())}")

    # If nothing safe, return any move to avoid crash
    if not safe_moves:
        print("  WARNING: No safe moves! Picking 'up' as last resort")
        return "up"

    # STEP 2: AVOID HEAD-TO-HEAD COLLISIONS WITH LARGER/EQUAL SNAKES
    safer_moves = {}
    for move, pos in safe_moves.items():
        if pos not in danger_zones:
            safer_moves[move] = pos
    
    # Only use safer moves if we have them
    if safer_moves:
        safe_moves = safer_moves
        print(f"  After avoiding head-to-head: {list(safe_moves.keys())}")

    # STEP 3: USE A* TO FIND PATH TO FOOD (when health requires it)
    astar_move = None
    if closest_food and my_health < food_threshold:
        path = a_star_pathfind(head, closest_food, game_state, blocked, hazards)
        if path:
            astar_move = get_direction_from_path(head, path)
            print(f"  A* found path to food: {len(path)} steps, next move: {astar_move}")
            
            # Verify A* move is in our safe moves
            if astar_move and astar_move in safe_moves:
                # Check if the path has enough space
                next_pos = safe_moves[astar_move]
                space_at_next = flood_fill(game_state, next_pos, blocked)
                min_space_needed = int((my_length // 2 + 3) * space_requirement_multiplier)
                
                # Use A* move if it has sufficient space OR if health is critical
                if space_at_next >= min_space_needed or my_health < 25:
                    print(f"  ✅ CHOSEN (A*): {astar_move} (space: {space_at_next})")
                    return astar_move
                else:
                    print(f"  ❌ A* move rejected: insufficient space ({space_at_next} < {min_space_needed})")

    # STEP 4: EVALUATE EACH MOVE WITH SCORING
    best_move = None
    best_score = -1
    
    for move, pos in safe_moves.items():
        # Calculate space available using flood fill
        space = flood_fill(game_state, pos, blocked)
        
        # Require minimum space based on our length and mode
        min_space_needed = int((my_length // 2 + 3) * space_requirement_multiplier)
        
        # Start with space as base score
        score = space * 10  # Weight space heavily
        
        # CRITICAL: Reject moves with insufficient space
        if space < min_space_needed:
            score -= 500
        
        # A* BONUS: If this is the A* suggested move, give it a boost
        if move == astar_move:
            score += 50
        
        # FOOD SEEKING: Adjusted based on growth phase and mode
        if closest_food and my_health < food_threshold:
            food_dist = manhattan_distance(pos, closest_food)
            
            if is_growth_phase and seek_food_aggressively:
                # GROWTH MODE: Heavily prioritize food
                if my_health < 30:
                    score += (250 - food_dist * 25)  # CRITICAL
                elif my_health < 60:
                    score += (200 - food_dist * 20)  # HIGH
                else:
                    score += (150 - food_dist * 15)  # NORMAL growth seeking
                
                # Extra bonus for being on the path to food
                if move == astar_move:
                    score += 100  # Double A* bonus in growth mode
                    
            elif is_dangerous_length:
                # DANGEROUS LENGTH: Avoid food unless critical
                if my_health < 30:
                    score += (100 - food_dist * 10)
                else:
                    score -= 20  # Slightly avoid food
                    
            else:
                # POST-GROWTH: Mode-specific behavior
                if mode == "defensive":
                    if my_health < 30:
                        score += (200 - food_dist * 20)
                    else:
                        score += (150 - food_dist * 15)
                elif mode == "aggressive":
                    if my_health < 30:
                        score += (100 - food_dist * 10)
                    else:
                        score += (30 - food_dist * 3)
                else:  # balanced
                    if my_health < 30:
                        score += (100 - food_dist * 10)
                    elif my_health < 60:
                        score += (50 - food_dist * 5)
                    else:
                        score += (25 - food_dist * 2)
        
        # OPPONENT INTERACTION (only after growth phase)
        if not is_growth_phase and opponents:
            if mode == "aggressive":
                # Seek confrontation with smaller snakes
                for opp in opponents:
                    if my_length >= opp["length"]:
                        dist_to_opponent = manhattan_distance(pos, opp["head"])
                        
                        if 2 <= dist_to_opponent <= 3:
                            score += 40  # Perfect hunting position
                        elif dist_to_opponent < 2:
                            score += 20
                        elif dist_to_opponent <= 5:
                            score += 10
                        
                        if opp["is_desperate"] and dist_to_opponent <= 4:
                            score += 30
            
            elif mode == "defensive":
                # Stay away from opponents
                for opp in opponents:
                    dist_to_opponent = manhattan_distance(pos, opp["head"])
                    
                    if dist_to_opponent < 3:
                        score -= 60
                    elif dist_to_opponent < 5:
                        score -= 30
                    elif dist_to_opponent > 7:
                        score += 10
        
        # AVOID EDGES: Less penalty in growth mode (need to reach food)
        edge_penalty = 0
        if pos[0] == 0 or pos[0] == width - 1:
            edge_penalty += 2 if is_growth_phase else 5
        if pos[1] == 0 or pos[1] == height - 1:
            edge_penalty += 2 if is_growth_phase else 5
        score -= edge_penalty
        
        # CENTER CONTROL: Only matters in mid/late game
        if phase in ["mid", "late"] and not is_growth_phase:
            center_x = width // 2
            center_y = height // 2
            dist_to_center = manhattan_distance(pos, (center_x, center_y))
            max_dist = manhattan_distance((0, 0), (center_x, center_y))
            center_score = (1 - dist_to_center / max_dist) * 15
            score += center_score
        
        print(f"  {move}: space={space}, food_dist={manhattan_distance(pos, closest_food) if closest_food else 'N/A'}, score={score:.1f}")
        
        if score > best_score:
            best_score = score
            best_move = move

    print(f"  ✅ CHOSEN: {best_move} (score: {best_score:.1f})")
    return best_move if best_move else list(safe_moves.keys())[0]


# --------------------------
# Battlesnake API
# --------------------------

def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "",
        "color": "#22aa66",
        "head": "safe",
        "tail": "bolt",
    }

def start(game_state: typing.Dict):
    print("GAME START")

def end(game_state: typing.Dict):
    print("GAME OVER\n")

def move(game_state: typing.Dict) -> typing.Dict:
    next_move = choose_move(game_state)
    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}

# --------------------------
# Server
# --------------------------

if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})

