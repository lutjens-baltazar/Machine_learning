import numpy as np
import random
import os

# Funciones auxiliares que pueden resultar útiles para tu implementación
from utils import bfs_search, get_valid_moves

# Path actual de trabajo
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

# Hiperparámetros de entrenamiento (jugar con ellos, estudiar que ocurre al cambiarlos)
CAT_MAX_EXPLORATION_RATE = 1
CAT_MIN_EXPLORATION_RATE = 0.0001
CAT_EXPLORATION_DECAY_RATE = 0.0001
CAT_LR = 0.1
CAT_DISCOUNT_RATE = 0.1

MOUSE_MAX_EXPLORATION_RATE = 1
MOUSE_MIN_EXPLORATION_RATE = 0.0001
MOUSE_EXPLORATION_DECAY_RATE = 0.0001
MOUSE_LR = 0.1
MOUSE_DISCOUNT_RATE = 0.1

class ReinforcedAgent:

    def __init__(self, position, table_name = None):

        # Posición inicial del agente
        self.pos = position

        # ===== CONSTRUCCIÓN DE LA Q-TABLE ===== #
        # Cargamos el mapa y buscamos las posiciones libres dentro de este
        free_positions = []
        lab_map = np.load(os.path.join(CURRENT_PATH, "game_map.npy"))
        for x in range(lab_map.shape[0]):
            for y in range(lab_map.shape[1]):
                if lab_map[x, y] == 0:
                    free_positions.append((x, y))

        # Diccionario que recibe una tupla del estado de juego de key y retorna el índice de su fila asociada en la Q-Table
        self.states_index = dict()
        index = 0
        for cat_pos in free_positions:
            for mouse_pos in free_positions:
                self.states_index[tuple([cat_pos[0], cat_pos[1], mouse_pos[0], mouse_pos[1]])] = index
                index += 1
            
        # Tasa de exploración del agente
        self.exploration_rate = 1
        
        # En caso de haber una Q-Table preexistente, utilizarla
        if table_name is None:
            self.q_table = np.zeros((index, 5))
        
        # En caso de no entregar una Q-Table, crear una llena de ceros
        else:
            self.q_table = np.load(os.path.join(CURRENT_PATH, "data", table_name))


    # Obtener la acción a ejecutar dado el estado del juego
    def get_action(self, lab_map, cat_pos, mouse_pos, noise = 0, train = False):

        # Si entrenamos, considerar si explorar o explotar
        if train:
            if random.random() < self.exploration_rate:
                return random.randint(0, 4)
        
        # Si no, considerar los movimientos como ruidosos, con una probabilidad de hacer uno aleatorio
        else:
            if random.random() < noise:
                return random.randint(0, 4)
        
        # Calculamos el estado actual del juego
        state = (cat_pos[0], cat_pos[1], mouse_pos[0], mouse_pos[1])


        # ===== COMPLETAR =====
        # Se debe retornar el movimiento que lleve a un mejor estado futuro, basándose en la Q-Table
        state_index = self.states_index[state]
        
        q_values = self.q_table[state_index]
        
        valid_moves = get_valid_moves(lab_map, cat_pos, mouse_pos)

        # Choose the action with the highest Q-value among the valid moves
        valid_q_values = [q_values[move] for move in valid_moves]
        move = valid_moves[np.argmax(valid_q_values)]
        return move
            
    
    def get_reward(self, lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos):
        pass
    
    def update_policy(self, lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos):
        state = (old_cat_pos[0], old_cat_pos[1], old_mouse_pos[0], old_mouse_pos[1])
        new_state = (new_cat_pos[0], new_cat_pos[1], new_mouse_pos[0], new_mouse_pos[1])
        
        reward = self.get_reward(lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos)

        # ===== COMPLETAR =====
        # Se debe actualizar el valor asociado al par estado-acción en la Q-Table
        # recuerda que la acción jugada fue action en el estado state
         # Get the old Q-value
        old_q_value = self.q_table[self.states_index[state], action]

        # Get the maximum Q-value for the new state
        max_new_q_value = np.max(self.q_table[self.states_index[new_state]])

        # Calculate the new Q-value
        new_q_value = old_q_value + CAT_LR * (reward + CAT_DISCOUNT_RATE * max_new_q_value - old_q_value)

        # Update the Q-table
        self.q_table[self.states_index[state], action] = new_q_value
        # =====================
    
    def update_exploration(self, n_game):
        pass

class RLCat(ReinforcedAgent):

    def __init__(self, position, table_path = None):

        super().__init__(position, table_path)
    def get_action(self, lab_map, cat_pos, mouse_pos, noise = 0, train = False):

        # Si entrenamos, considerar si explorar o explotar
        if train:
            if random.random() < self.exploration_rate:
                return random.randint(0, 4)
        
        # Si no, considerar los movimientos como ruidosos, con una probabilidad de hacer uno aleatorio
        else:
            if random.random() < noise:
                return random.randint(0, 4)
        # Calculate the current game state
        state = (cat_pos[0], cat_pos[1], mouse_pos[0], mouse_pos[1])

        # Get the index of the current state in the Q-table
        state_index = self.states_index[state]

        # Get the Q-values for the current state
        q_values = self.q_table[state_index]

        # Get the valid moves for the cat
        valid_moves = get_valid_moves(lab_map, cat_pos)
        # Choose the action with the highest Q-value among the valid moves
        valid_q_values = [q_values[move] for move in valid_moves]
        move = valid_moves[np.argmax(valid_q_values)]
        return move
            
            
            
    def get_reward(self, lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos):
        # Calculate the BFS distance between the cat and the mouse
        old_distance = len(bfs_search(lab_map, old_cat_pos, old_mouse_pos)) if bfs_search(lab_map, old_cat_pos, old_mouse_pos) else float('inf')
        new_distance = len(bfs_search(lab_map, new_cat_pos, new_mouse_pos)) if bfs_search(lab_map, new_cat_pos, new_mouse_pos) else float('inf')

        # Reward the cat for getting closer to the mouse
        if new_distance < old_distance:
            reward = 1
        # Punish the cat for getting further away from the mouse
        elif new_distance > old_distance:
            reward = -1
        # No reward or punishment for staying the same distance from the mouse
        else:
            reward = 0

        # Extra reward for catching the mouse
        if new_distance == 0:
            reward += 10

        return reward
    
    def update_exploration(self, n_game):
        # Start with a high exploration rate and gradually decrease it over time
        self.exploration_rate = max(CAT_MIN_EXPLORATION_RATE, CAT_MAX_EXPLORATION_RATE * np.exp(-CAT_EXPLORATION_DECAY_RATE * n_game))

        # Cada 1000 partidas, aprovecharemos de guardar la tabla de desempeño del agente
        if n_game % 1000 == 0:
            np.save(os.path.join(CURRENT_PATH, "data", f"QTableCat{n_game}.npy"), self.q_table)
            print(f"Epsilon: {self.exploration_rate} | Guardando QTable en agents/data/QTableCat{n_game}.npy")
    
class RLMouse(ReinforcedAgent):
    def __init__(self, position, table_path = None):

        super().__init__(position, table_path)
    def get_action(self, lab_map, cat_pos, mouse_pos, noise = 0, train = False):

        # Si entrenamos, considerar si explorar o explotar
        if train:
            if random.random() < self.exploration_rate:
                return random.randint(0, 4)
        
        # Si no, considerar los movimientos como ruidosos, con una probabilidad de hacer uno aleatorio
        else:
            if random.random() < noise:
                return random.randint(0, 4)
        # Calculate the current game state
        state = (cat_pos[0], cat_pos[1], mouse_pos[0], mouse_pos[1])

        # Get the index of the current state in the Q-table
        state_index = self.states_index[state]

        # Get the Q-values for the current state
        q_values = self.q_table[state_index]

        # Get the valid moves for the cat
        valid_moves = get_valid_moves(lab_map, mouse_pos)
        # Choose the action with the highest Q-value among the valid moves
        valid_q_values = [q_values[move] for move in valid_moves]
        move = valid_moves[np.argmax(valid_q_values)]
        return move
            
    def get_reward(self, lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos):
        # Calculate the BFS distance between the cat and the mouse
        old_distance = len(bfs_search(lab_map, old_mouse_pos, old_cat_pos)) if bfs_search(lab_map, old_mouse_pos, old_cat_pos) else float('inf')
        new_distance = len(bfs_search(lab_map, new_mouse_pos, new_cat_pos)) if bfs_search(lab_map, new_mouse_pos, new_cat_pos) else float('inf')

        # Reward the mouse for getting further from the cat
        if new_distance > old_distance:
            reward = 1
        # Punish the mouse for getting closer to the cat
        elif new_distance < old_distance:
            reward = -1
        # No reward or punishment for staying the same distance from the cat
        else:
            reward = 0

        # Extra punishment for getting caught by the cat
        if new_distance == 0:
            reward -= 10

        return reward
    
    def update_exploration(self, n_game):
        # ===== COMPLETAR =====
        # Se debe actualizar la tasa de exploración del agente
        self.exploration_rate = max(MOUSE_MIN_EXPLORATION_RATE, MOUSE_MAX_EXPLORATION_RATE * np.exp(-MOUSE_EXPLORATION_DECAY_RATE * n_game))
        # =====================

        # Cada 1000 partidas, aprovecharemos de guardar la tabla de desempeño del agente
        if n_game % 1000 == 0:
            np.save(os.path.join(CURRENT_PATH, "data", f"QTableMouse{n_game}.npy"), self.q_table)
            print(f"Epsilon: {self.exploration_rate} | Guardando QTable en agents/data/QTableMouse{n_game}.npy")
