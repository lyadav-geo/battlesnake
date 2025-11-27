import random
import typing
from collections import deque
import heapq
import math


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

def get_next_position(head: typing.Dict, direction: str) -> typing.Dict:
    """Calculate next position given current head and direction"""
    x, y = head["x"], head["y"]
    if direction == "up":
        return {"x": x, "y": y + 1}
    elif direction == "down":
        return {"x": x, "y": y - 1}
    elif direction == "left":
        return {"x": x - 1, "y": y}
    elif direction == "right":
        return {"x": x + 1, "y": y}
    return head


def is_out_of_bounds(pos: typing.Dict, board_width: int, board_height: int) -> bool:
    """Check if position is outside board boundaries"""
    return pos["x"] < 0 or pos["x"] >= board_width or pos["y"] < 0 or pos["y"] >= board_height


def count_adjacent_walls(pos: typing.Dict, board_width: int, board_height: int) -> int:
    """Count how many walls are adjacent to this position"""
    walls = 0
    if pos["x"] == 0:
        walls += 1
    if pos["x"] == board_width - 1:
        walls += 1
    if pos["y"] == 0:
        walls += 1
    if pos["y"] == board_height - 1:
        walls += 1
    return walls


def count_safe_neighbors(pos: typing.Dict, game_state: typing.Dict, snake_positions: set) -> int:
    """Count how many safe adjacent squares exist from this position"""
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]

    directions = ["up", "down", "left", "right"]
    safe_count = 0

    for direction in directions:
        next_pos = get_next_position(pos, direction)

        if is_out_of_bounds(next_pos, board_width, board_height):
            continue
        if (next_pos["x"], next_pos["y"]) in snake_positions:
            continue

        safe_count += 1

    return safe_count


def manhattan_distance(pos1: typing.Dict, pos2: typing.Dict) -> int:
    """Calculate Manhattan distance between two positions"""
    return abs(pos1["x"] - pos2["x"]) + abs(pos1["y"] - pos2["y"])


def astar_distance(start: typing.Dict, goal: typing.Dict, game_state: typing.Dict, snake_positions: set) -> typing.Optional[int]:
    """
    Calculate actual path distance using A* pathfinding.
    Returns path length or None if no path exists.
    """
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]

    # Priority queue: (f_score, g_score, position)
    open_set = []
    heapq.heappush(open_set, (0, 0, (start["x"], start["y"])))

    # Track best g_score for each position
    g_scores = {(start["x"], start["y"]): 0}

    # Track visited nodes
    closed_set = set()

    directions = ["up", "down", "left", "right"]

    while open_set:
        f_score, g_score, current_tuple = heapq.heappop(open_set)
        current = {"x": current_tuple[0], "y": current_tuple[1]}

        # Check if we reached the goal
        if current["x"] == goal["x"] and current["y"] == goal["y"]:
            return g_score

        # Skip if already visited
        if current_tuple in closed_set:
            continue

        closed_set.add(current_tuple)

        # Explore neighbors
        for direction in directions:
            neighbor = get_next_position(current, direction)
            neighbor_tuple = (neighbor["x"], neighbor["y"])

            # Skip if out of bounds
            if is_out_of_bounds(neighbor, board_width, board_height):
                continue

            # Skip if hits a snake body (but allow goal position)
            if neighbor_tuple in snake_positions:
                if neighbor["x"] != goal["x"] or neighbor["y"] != goal["y"]:
                    continue

            # Calculate tentative g_score
            tentative_g = g_score + 1

            # Skip if we've found a better path to this neighbor
            if neighbor_tuple in g_scores and tentative_g >= g_scores[neighbor_tuple]:
                continue

            # This is the best path to neighbor so far
            g_scores[neighbor_tuple] = tentative_g

            # Calculate f_score (g + heuristic)
            h_score = manhattan_distance(neighbor, goal)
            f_score = tentative_g + h_score

            heapq.heappush(open_set, (f_score, tentative_g, neighbor_tuple))

    # No path found
    return None


def flood_fill(start: typing.Dict, game_state: typing.Dict, avoid_positions: set) -> int:
    """Count reachable spaces from start position using BFS"""
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]

    visited = set()
    queue = deque([start])
    visited.add((start["x"], start["y"]))

    directions = ["up", "down", "left", "right"]

    while queue:
        current = queue.popleft()

        for direction in directions:
            next_pos = get_next_position(current, direction)
            pos_tuple = (next_pos["x"], next_pos["y"])

            if pos_tuple in visited:
                continue
            if is_out_of_bounds(next_pos, board_width, board_height):
                continue
            if pos_tuple in avoid_positions:
                continue

            visited.add(pos_tuple)
            queue.append(next_pos)

    return len(visited)


def calculate_voronoi(game_state: typing.Dict, snake_positions: set) -> typing.Dict:
    """
    Calculate Voronoi territories - which cells each snake can reach first.
    Returns dict with snake IDs as keys and territory counts as values.
    """
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]

    # Track which snake reaches each cell first and at what distance
    cell_owner = {}  # (x, y) -> snake_id
    cell_distance = {}  # (x, y) -> distance

    # Initialize BFS from all snake heads simultaneously
    queue = deque()
    for snake in game_state["board"]["snakes"]:
        snake_id = snake.get("id")
        if not snake_id or len(snake.get("body", [])) == 0:
            continue

        head = snake["body"][0]
        head_tuple = (head["x"], head["y"])
        queue.append((head_tuple, snake_id, 0))
        cell_owner[head_tuple] = snake_id
        cell_distance[head_tuple] = 0

    directions = ["up", "down", "left", "right"]

    # BFS to fill the board
    while queue:
        current_tuple, snake_id, distance = queue.popleft()
        current = {"x": current_tuple[0], "y": current_tuple[1]}

        for direction in directions:
            next_pos = get_next_position(current, direction)
            next_tuple = (next_pos["x"], next_pos["y"])

            # Skip out of bounds
            if is_out_of_bounds(next_pos, board_width, board_height):
                continue

            # Skip snake bodies
            if next_tuple in snake_positions:
                continue

            new_distance = distance + 1

            # If unclaimed or we can reach it faster/same time
            if next_tuple not in cell_distance or new_distance <= cell_distance[next_tuple]:
                # If same distance, it's contested (could mark as neutral, but we'll give it to first)
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


def get_game_phase(game_state: typing.Dict) -> str:
    """Determine current game phase: early, late, or 1v1"""
    alive_snakes = len(game_state["board"]["snakes"])

    if alive_snakes == 2:
        return "1v1"
    elif alive_snakes <= 3:
        return "late"
    else:
        return "early"


def get_opponent_heads(game_state: typing.Dict) -> typing.List:
    """Get positions and lengths of opponent snake heads"""
    my_id = game_state["you"].get("id", "")
    opponent_heads = []

    for snake in game_state["board"]["snakes"]:
        snake_id = snake.get("id", "")
        if snake_id != my_id and len(snake.get("body", [])) > 0:
            opponent_heads.append({
                "pos": snake["body"][0],
                "length": len(snake["body"]),
                "id": snake_id
            })

    return opponent_heads


def get_opponent_stats(game_state: typing.Dict) -> typing.Dict:
    """Get stats about opponents: max length, avg length, count"""
    my_id = game_state["you"].get("id", "")
    opponent_lengths = []

    for snake in game_state["board"]["snakes"]:
        snake_id = snake.get("id", "")
        if snake_id != my_id and len(snake.get("body", [])) > 0:
            opponent_lengths.append(len(snake["body"]))

    if len(opponent_lengths) == 0:
        return {"max_length": 0, "avg_length": 0, "count": 0}

    return {
        "max_length": max(opponent_lengths),
        "avg_length": sum(opponent_lengths) / len(opponent_lengths),
        "count": len(opponent_lengths)
    }


def is_head_to_head_safe(pos: typing.Dict, my_length: int, opponent_heads: typing.List) -> bool:
    """Check if position is safe from head-to-head collisions"""
    for opponent in opponent_heads:
        if manhattan_distance(pos, opponent["pos"]) <= 2:
            if opponent["length"] >= my_length:
                return False
    return True


def get_all_snake_positions(game_state: typing.Dict) -> set:
    """Get all positions occupied by snake bodies that will still be there next turn"""
    positions = set()

    for snake in game_state["board"]["snakes"]:
        body = snake["body"]

        if len(body) == 0:
            continue

        # Snake just ate if the last two body segments are at the same position
        # This happens when the tail duplicated on the previous turn
        snake_just_ate = len(body) > 1 and body[-1]["x"] == body[-2]["x"] and body[-1]["y"] == body[-2]["y"]

        for i, segment in enumerate(body):
            # Skip head (it will move to a new position) - index 0
            if i == 0:
                continue

            # Skip tail UNLESS snake just ate (then tail stays) - last index
            if i == len(body) - 1 and not snake_just_ate:
                continue

            # Add this body segment as an obstacle
            positions.add((segment["x"], segment["y"]))

    return positions


def move(game_state: typing.Dict) -> typing.Dict:
    my_head = game_state["you"]["body"][0]
    my_length = len(game_state["you"]["body"])
    my_health = game_state["you"]["health"]
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    food = game_state["board"]["food"]

    print(f"Turn {game_state['turn']}: Head at ({my_head['x']}, {my_head['y']}), Board: {board_width}x{board_height}")

    game_phase = get_game_phase(game_state)
    opponent_heads = get_opponent_heads(game_state)
    opponent_stats = get_opponent_stats(game_state)

    # Get all snake body positions that will still be there next turn
    snake_positions = get_all_snake_positions(game_state)

    my_body = [(seg["x"], seg["y"]) for seg in game_state["you"]["body"]]
    length_target = opponent_stats['max_length'] + 2
    need_length = my_length < length_target

    # Debug: log all snake bodies
    print(f"  Total obstacle positions: {len(snake_positions)}")
    print(f"  Obstacle positions set: {sorted(list(snake_positions))[:20]}...")  # Show first 20
    for snake in game_state["board"]["snakes"]:
        snake_id = snake.get("id", "unknown")
        snake_name = snake.get("name", snake_id[:8] if snake_id else "unknown")
        is_me = snake_id == game_state["you"].get("id", "")
        body_positions = [(seg["x"], seg["y"]) for seg in snake.get("body", [])]
        print(f"    {'MY' if is_me else 'OPP'} {snake_name}: length {len(snake.get('body', []))}, body {body_positions}")

    # Calculate current Voronoi territories
    my_id = game_state["you"].get("id", "")
    voronoi_territories = calculate_voronoi(game_state, snake_positions)
    my_territory = voronoi_territories.get(my_id, 0)

    # Calculate our territory share
    total_territory = sum(voronoi_territories.values())
    my_territory_pct = (my_territory / total_territory * 100) if total_territory > 0 else 0

    print(f"  My length: {my_length}, Target: {length_target}, Max opponent: {opponent_stats['max_length']}, Avg: {opponent_stats['avg_length']:.1f}")
    print(f"  Phase: {game_phase}, Health: {my_health}, Need length: {need_length}")
    print(f"  My territory: {my_territory} cells ({my_territory_pct:.1f}%)")

    # Show opponent territories
    for snake in game_state["board"]["snakes"]:
        snake_id = snake.get("id", "")
        if snake_id and snake_id != my_id:
            opp_territory = voronoi_territories.get(snake_id, 0)
            opp_pct = (opp_territory / total_territory * 100) if total_territory > 0 else 0
            snake_name = snake.get("name", snake_id[:10])
            print(f"    Opponent {snake_name[:10]}: {opp_territory} cells ({opp_pct:.1f}%), length {len(snake.get('body', []))}")
    if len(food) > 0:
        # Find closest food by actual path distance
        food_with_dist = []
        for f in food:
            astar_dist = astar_distance(my_head, f, game_state, snake_positions)
            if astar_dist is not None:
                food_with_dist.append((f, astar_dist))

        if food_with_dist:
            closest_food, closest_dist = min(food_with_dist, key=lambda x: x[1])
            print(f"  Closest food at ({closest_food['x']}, {closest_food['y']}), A* distance: {closest_dist}")

    directions = ["up", "down", "left", "right"]
    move_scores = {}

    # Prevent reversing direction into neck
    my_neck = game_state["you"]["body"][1] if len(game_state["you"]["body"]) > 1 else None

    for direction in directions:
        next_pos = get_next_position(my_head, direction)
        score = 0

        # SURVIVAL CHECKS - immediate disqualification

        # Check reversing into neck
        if my_neck is not None:
            if next_pos["x"] == my_neck["x"] and next_pos["y"] == my_neck["y"]:
                print(f"  {direction}: REVERSE INTO NECK at ({next_pos['x']}, {next_pos['y']})")
                move_scores[direction] = -10000
                continue

        # Check walls
        if is_out_of_bounds(next_pos, board_width, board_height):
            print(f"  {direction}: OUT OF BOUNDS ({next_pos['x']}, {next_pos['y']}) board: {board_width}x{board_height}")
            move_scores[direction] = -10000
            continue

        # Check snake bodies
        if (next_pos["x"], next_pos["y"]) in snake_positions:
            # Find which snake we would collide with
            colliding_snake = "unknown"
            for snake in game_state["board"]["snakes"]:
                for segment in snake.get("body", []):
                    if segment["x"] == next_pos["x"] and segment["y"] == next_pos["y"]:
                        snake_id = snake.get("id", "unknown")
                        colliding_snake = snake.get("name", snake_id[:8] if snake_id else "unknown")
                        break
                if colliding_snake != "unknown":
                    break
            print(f"  {direction}: SNAKE COLLISION at ({next_pos['x']}, {next_pos['y']}) with {colliding_snake}")
            move_scores[direction] = -10000
            continue

        # Check head-to-head collisions
        if not is_head_to_head_safe(next_pos, my_length, opponent_heads):
            print(f"  {direction}: HEAD-TO-HEAD RISK at ({next_pos['x']}, {next_pos['y']})")
            move_scores[direction] = -10000
            continue

        # TRAP DETECTION - check if we're walking into a corner or tight spot
        adjacent_walls = count_adjacent_walls(next_pos, board_width, board_height)
        safe_neighbors = count_safe_neighbors(next_pos, game_state, snake_positions)

        # Corner = 2 walls, tight spot = limited safe neighbors
        if adjacent_walls >= 2:
            print(f"  {direction}: CORNER ({adjacent_walls} walls)")
            score -= 1500

        if safe_neighbors <= 1:
            # Only 1 or 0 escape routes from this position
            print(f"  {direction}: TRAP ({safe_neighbors} safe neighbors)")
            score -= 1000
        elif safe_neighbors == 2:
            # Penalize but not as severely
            score -= 200

        # SPACE CONTROL - flood fill to count reachable spaces
        avoid_positions = snake_positions.copy()
        reachable_spaces = flood_fill(next_pos, game_state, avoid_positions)

        # Must have enough space to survive
        if reachable_spaces < my_length:
            score -= 5000
        else:
            # Determine if we need to grow
            length_target = opponent_stats['max_length'] + 2
            need_length = my_length < length_target

            # Reduce space control weight in early game when we need food
            # This prevents circling in open space instead of eating
            if game_phase == "early" and need_length:
                score += reachable_spaces * 1  # Minimal weight when growth is priority
            elif game_phase == "early":
                score += reachable_spaces * 3  # Low weight in early game
            else:
                score += reachable_spaces * 10  # Normal weight for late game

        # VORONOI TERRITORY CONTROL - calculate territory if we make this move
        # Create a simulated game state with our head at next_pos
        simulated_game_state = {
            "board": {
                "width": board_width,
                "height": board_height,
                "snakes": []
            }
        }

        # Add our snake with new head position
        simulated_my_body = [next_pos] + game_state["you"]["body"][:-1]  # Move head, remove tail
        simulated_game_state["board"]["snakes"].append({
            "id": my_id,
            "body": simulated_my_body
        })

        # Add opponent snakes (unchanged for now - simplified simulation)
        for snake in game_state["board"]["snakes"]:
            snake_id = snake.get("id", "")
            if snake_id and snake_id != my_id:
                simulated_game_state["board"]["snakes"].append(snake)

        # Calculate voronoi after this move
        simulated_snake_positions = get_all_snake_positions(simulated_game_state)
        future_voronoi = calculate_voronoi(simulated_game_state, simulated_snake_positions)
        future_my_territory = future_voronoi.get(my_id, 0)

        # Territory gain/loss
        territory_change = future_my_territory - my_territory

        # Calculate how much we're cutting off opponents
        opponent_territory_loss = 0
        for opp_id, opp_territory in voronoi_territories.items():
            if opp_id and opp_id != my_id:
                future_opp_territory = future_voronoi.get(opp_id, 0)
                opponent_territory_loss += (opp_territory - future_opp_territory)

        # Reward territory gain, especially in late game
        if game_phase in ["late", "1v1"]:
            score += territory_change * 15  # Strong weight in late game
            score += opponent_territory_loss * 10  # Reward cutting off opponents
        elif game_phase == "early" and not need_length:
            score += territory_change * 5  # Moderate in early game if already long enough
            score += opponent_territory_loss * 3

        # Debug log for territory changes
        if territory_change != 0 or opponent_territory_loss > 0:
            print(f"  {direction}: Territory change: {territory_change:+d}, Opponent loss: {opponent_territory_loss}")

        # FOOD SEEKING - aggressive early growth, strategic late game
        if len(food) > 0:
            # Use A* to find actual path distance to each food
            food_distances = []
            for f in food:
                astar_dist = astar_distance(next_pos, f, game_state, snake_positions)
                if astar_dist is not None:
                    food_distances.append(astar_dist)
                else:
                    # No path to this food, use manhattan as penalty
                    food_distances.append(manhattan_distance(next_pos, f) + 50)

            closest_food_dist = min(food_distances) if food_distances else 999

            # Determine if we need to grow
            length_target = opponent_stats['max_length'] + 2
            need_length = my_length < length_target

            # Reduce food seeking if this move is into a trap (let trap penalty dominate)
            # This prevents suiciding for food in corners
            food_weight_multiplier = 1.0
            if adjacent_walls >= 2 or safe_neighbors <= 1:
                food_weight_multiplier = 0.5  # Reduce food attraction by 50%

            # EARLY GAME: aggressive growth until we have length advantage
            if game_phase == "early":
                if need_length:
                    # Very aggressive food seeking when behind in length
                    score -= closest_food_dist * 100 * food_weight_multiplier
                    score += (100 - my_health) * 8
                    # Extra bonus for being shorter than opponents
                    if my_length <= opponent_stats['avg_length']:
                        score -= closest_food_dist * 60 * food_weight_multiplier
                elif my_health < 50:
                    # Have length advantage but need health
                    score -= closest_food_dist * 80 * food_weight_multiplier
                    score += (100 - my_health) * 5
                else:
                    # Have length advantage and good health - moderate food seeking
                    score -= closest_food_dist * 30 * food_weight_multiplier

            # LATE GAME: only seek food when necessary
            elif game_phase in ["late", "1v1"]:
                if my_health < 35:
                    # Desperate for food
                    score -= closest_food_dist * 60 * food_weight_multiplier
                    score += (100 - my_health) * 8
                elif need_length and my_health < 60:
                    # Need length and health getting low
                    score -= closest_food_dist * 40 * food_weight_multiplier
                elif my_health > 70:
                    # Healthy - avoid food to focus on space control and aggression
                    score += closest_food_dist * 3

        # AGGRESSION - context-dependent based on advantages
        for opponent in opponent_heads:
            opponent_dist = manhattan_distance(next_pos, opponent["pos"])
            opponent_id = opponent.get("id", "")
            opponent_territory = voronoi_territories.get(opponent_id, 0)

            # Determine if we have territorial advantage
            territory_advantage = my_territory > opponent_territory

            # Late game and 1v1: strong aggression with advantages
            if game_phase in ["late", "1v1"]:
                # If we're longer OR have territory advantage, be aggressive
                if my_length > opponent["length"] or territory_advantage:
                    score -= opponent_dist * 5

                    # Extra aggressive in 1v1
                    if game_phase == "1v1":
                        score -= opponent_dist * 10

                    # Even more aggressive if we have both length AND territory advantage
                    if my_length > opponent["length"] and territory_advantage:
                        score -= opponent_dist * 8

                # If we're smaller AND losing territory, maintain distance
                elif my_length < opponent["length"] and not territory_advantage:
                    if opponent_dist < 3:
                        score += opponent_dist * 15

            # Early game: mild aggression if we have strong territory advantage
            elif game_phase == "early":
                # If we have huge territory advantage, push them a bit
                if territory_advantage and my_territory > opponent_territory * 1.5:
                    score -= opponent_dist * 2

                # If they're getting too close to our food/space, push back
                if opponent_dist < 4 and territory_advantage:
                    score -= opponent_dist * 1

        # CENTER CONTROL - early game (lower priority than food)
        if game_phase == "early" and my_health > 60:
            # Only care about center when healthy (food is more important)
            center_x = board_width // 2
            center_y = board_height // 2
            center_dist = abs(next_pos["x"] - center_x) + abs(next_pos["y"] - center_y)
            score -= center_dist * 1

        # PREFER OPEN SPACE - bonus for moves with more escape routes
        # But only apply this when not aggressively seeking food
        if safe_neighbors >= 3 and not (game_phase == "early" and need_length):
            score += 50  # Good position with multiple exits

        move_scores[direction] = score

    # Select best move - prioritize avoiding walls and neck absolutely
    definitely_safe_moves = []
    for direction in directions:
        next_pos = get_next_position(my_head, direction)
        # Must not be out of bounds
        if is_out_of_bounds(next_pos, board_width, board_height):
            continue
        # Must not be reversing into neck
        if my_neck is not None and next_pos["x"] == my_neck["x"] and next_pos["y"] == my_neck["y"]:
            continue
        definitely_safe_moves.append(direction)

    print(f"  Move scores: {move_scores}")
    print(f"  Definitely safe (not walls/neck): {definitely_safe_moves}")

    # Filter to moves that are both safe AND have good scores
    safe_moves = [move for move in definitely_safe_moves if move_scores[move] > -10000]

    print(f"  Safe moves (safe + good score): {safe_moves}")

    if len(safe_moves) == 0:
        # Desperate: choose best safe move even if it hits a snake
        if len(definitely_safe_moves) > 0:
            print(f"MOVE {game_state['turn']}: Desperate! Choosing best non-wall/non-neck move")
            next_move = max(definitely_safe_moves, key=lambda m: move_scores[m])
        else:
            # Completely trapped - all moves are out of bounds
            # This should never happen, but pick the least-bad out-of-bounds move
            print(f"MOVE {game_state['turn']}: ALL MOVES OUT OF BOUNDS! Picking least bad")
            next_move = max(move_scores.keys(), key=lambda m: move_scores[m])
    else:
        # Choose move with highest score from safe moves
        next_move = max(safe_moves, key=lambda m: move_scores[m])

    # Final verification before returning
    final_pos = get_next_position(my_head, next_move)
    if is_out_of_bounds(final_pos, board_width, board_height):
        print(f"CRITICAL WARNING: Selected move {next_move} is OUT OF BOUNDS! Head: ({my_head['x']}, {my_head['y']}), Next: ({final_pos['x']}, {final_pos['y']}), Board: {board_width}x{board_height}")
        print(f"  This should never happen! Definitely safe moves: {definitely_safe_moves}")
        print(f"  All move scores: {move_scores}")

        # Emergency override: pick first definitely safe move or random if none
        if len(definitely_safe_moves) > 0:
            next_move = definitely_safe_moves[0]
            print(f"  EMERGENCY OVERRIDE: Choosing {next_move} instead")
        else:
            # Truly desperate - try all directions and pick first one that's in bounds
            for emergency_direction in ["up", "down", "left", "right"]:
                emergency_pos = get_next_position(my_head, emergency_direction)
                if not is_out_of_bounds(emergency_pos, board_width, board_height):
                    next_move = emergency_direction
                    print(f"  EMERGENCY OVERRIDE: Choosing {next_move} as last resort")
                    break

    print(f"MOVE {game_state['turn']}: {next_move} (phase: {game_phase}, health: {my_health}, score: {move_scores.get(next_move, 'N/A')})")
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})