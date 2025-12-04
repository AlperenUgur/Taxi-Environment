import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

Q_FILE = "taxi_qtable_6x6.npy"

class TaxiEnv6x6(gym.Env):
    """
    6x6 grid ortamı:
    - Taxi: her hücreye girebilir (yasak hücreler hariç)
    - Yolcu: her hücrede bekleyebilir (yasak hücreler hariç)
    - Hedef: her hücre olabilir (yasak hücreler hariç)
    - Yasak hücre: taksi de yolcu da giremez
    - State: (taxi_r, taxi_c, target_r, target_c, has_passenger)
      target = yolcu hücresi (yolcu takside değilken)
      target = hedef hücresi (yolcu taksideyken)
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.rows = 6
        self.cols = 6

        # ------------------ YASAK HÜCRELER ------------------
        self.forbidden_cells = {
            (1, 1),
            (2, 3),
            (4, 4),
        }

        # ------------------------ DUVARLAR ------------------------
        # Duvarlar: ((r1, c1), (r2, c2)) iki komşu hücre arası
        self.walls = {
            tuple(sorted(((1, 3), (1, 4)))),  # yatay
            tuple(sorted(((3, 1), (4, 1)))),  # dikey
            tuple(sorted(((3, 2), (4, 2)))),  # dikey
        }

        # State sayısı: taxi(36) * target(36) * has(2)
        self.n_states = (self.rows * self.cols) * (self.rows * self.cols) * 2

        # Eylemler: 0 up, 1 down, 2 left, 3 right, 4 pickup, 5 dropoff
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Discrete(self.n_states)

        # Render
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.render_episode_idx = 0
        self.render_step_idx = 0

        # Dinamik değişkenler
        self.taxi_row = None
        self.taxi_col = None
        self.pass_row = None
        self.pass_col = None
        self.dest_row = None
        self.dest_col = None
        self.has_passenger = None
        self.step_count = 0
        self.max_steps = 200

        # Salınım denetimi
        self.last_move_action = None   # sadece 0–3 için
        self.oscillation_penalty = -0.5

    # ------------ State encode/decode (int <-> koordinatlar) ------------

    def encode(self, taxi_r, taxi_c, target_r, target_c, has_passenger):
        """
        taxi_r, taxi_c: 0..5
        target_r, target_c: 0..5
        has_passenger: 0/1
        """
        i = taxi_r * self.cols + taxi_c                # 0..35
        i = i * (self.rows * self.cols) + (target_r * self.cols + target_c)  # target
        i = i * 2 + has_passenger
        return int(i)

    def decode(self, i):
        has_passenger = i % 2
        i //= 2
        target_idx = i % (self.rows * self.cols)
        i //= (self.rows * self.cols)
        taxi_idx = i

        target_r = target_idx // self.cols
        target_c = target_idx % self.cols

        taxi_r = taxi_idx // self.cols
        taxi_c = taxi_idx % self.cols

        return taxi_r, taxi_c, target_r, target_c, has_passenger

    # ---------------------- Gym API: reset ----------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        def random_free_cell():
            while True:
                r = rng.integers(0, self.rows)
                c = rng.integers(0, self.cols)
                if (r, c) not in self.forbidden_cells:
                    return r, c

        # Taxi
        self.taxi_row, self.taxi_col = random_free_cell()

        # Yolcu
        while True:
            self.pass_row, self.pass_col = random_free_cell()
            if (self.pass_row, self.pass_col) != (self.taxi_row, self.taxi_col):
                break

        # Hedef
        while True:
            self.dest_row, self.dest_col = random_free_cell()
            if (self.dest_row, self.dest_col) not in [
                (self.taxi_row, self.taxi_col),
                (self.pass_row, self.pass_col),
            ]:
                break

        self.has_passenger = 0
        self.step_count = 0
        self.last_move_action = None

        # Başlangıçta target = yolcu konumu
        target_r, target_c = self.pass_row, self.pass_col

        state = self.encode(
            self.taxi_row,
            self.taxi_col,
            target_r,
            target_c,
            self.has_passenger,
        )
        return state, {}

    # ---------------------- Gym API: step ----------------------

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        self.step_count += 1

        # -------- Hareket aksiyonları (0–3) --------
        if action in [0, 1, 2, 3]:
            new_r, new_c = self.taxi_row, self.taxi_col

            if action == 0:  # up
                new_r = max(self.taxi_row - 1, 0)
                edge = ((self.taxi_row, self.taxi_col),
                        (self.taxi_row - 1, self.taxi_col))
            elif action == 1:  # down
                new_r = min(self.taxi_row + 1, self.rows - 1)
                edge = ((self.taxi_row, self.taxi_col),
                        (self.taxi_row + 1, self.taxi_col))
            elif action == 2:  # left
                new_c = max(self.taxi_col - 1, 0)
                edge = ((self.taxi_row, self.taxi_col),
                        (self.taxi_row, self.taxi_col - 1))
            else:  # right
                new_c = min(self.taxi_col + 1, self.cols - 1)
                edge = ((self.taxi_row, self.taxi_col),
                        (self.taxi_row, self.taxi_col + 1))

            edge = tuple(sorted(edge))

            if edge in self.walls or (new_r, new_c) in self.forbidden_cells:
                # duvar/yasak: hareket yok, ekstra ceza
                reward += -2.0
            else:
                self.taxi_row, self.taxi_col = new_r, new_c
                reward += -1.0

            # salınım denetimi
            if self.last_move_action is not None:
                if (self.last_move_action == 0 and action == 1) or (
                    self.last_move_action == 1 and action == 0
                ):
                    reward += self.oscillation_penalty
                if (self.last_move_action == 2 and action == 3) or (
                    self.last_move_action == 3 and action == 2
                ):
                    reward += self.oscillation_penalty
            self.last_move_action = action

        # -------- Pickup (4) --------
        elif action == 4:
            if (
                self.has_passenger == 0
                and (self.taxi_row, self.taxi_col) == (self.pass_row, self.pass_col)
            ):
                self.has_passenger = 1
                reward += 5.0
            else:
                reward += -4.0

        # -------- Dropoff (5) --------
        elif action == 5:
            if (
                self.has_passenger == 1
                and (self.taxi_row, self.taxi_col) == (self.dest_row, self.dest_col)
            ):
                self.has_passenger = 0
                reward += 20.0
                terminated = True
            else:
                reward += -4.0

        # Çok uzayan epizod
        if self.step_count >= self.max_steps and not terminated:
            truncated = True
            reward += -10.0

        # Target konumu: yolcu taksideyse hedef, değilse yolcu
        if self.has_passenger == 0:
            target_r, target_c = self.pass_row, self.pass_col
        else:
            target_r, target_c = self.dest_row, self.dest_col

        state = self.encode(
            self.taxi_row,
            self.taxi_col,
            target_r,
            target_c,
            self.has_passenger,
        )

        return state, reward, terminated, truncated, {}

    # --------------------------- RENDER ---------------------------

    def _init_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.set_xlim(0, self.cols)
        self.ax.set_ylim(0, self.rows)
        self.ax.set_xticks(np.arange(0, self.cols + 1, 1))
        self.ax.set_yticks(np.arange(0, self.rows + 1, 1))
        self.ax.grid(True)
        self.ax.invert_yaxis()
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    def render(self, mode="rgb_array"):
        if self.fig is None or self.ax is None:
            self._init_figure()

        self.ax.clear()
        self.ax.set_xlim(0, self.cols)
        self.ax.set_ylim(0, self.rows)
        self.ax.set_xticks(np.arange(0, self.cols + 1, 1))
        self.ax.set_yticks(np.arange(0, self.rows + 1, 1))
        self.ax.grid(True, linewidth=0.5)
        self.ax.invert_yaxis()
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # Yasak hücreler (kırmızı)
        for (r, c) in self.forbidden_cells:
            rect = patches.Rectangle((c, r), 1, 1, facecolor="red", alpha=0.7)
            self.ax.add_patch(rect)

        # Hedef (yeşil)
        rect = patches.Rectangle(
            (self.dest_col, self.dest_row), 1, 1, facecolor="yellowgreen", alpha=0.7
        )
        self.ax.add_patch(rect)

        # Yolcu (sarı) – takside değilse
        if self.has_passenger == 0:
            rect = patches.Rectangle(
                (self.pass_col, self.pass_row), 1, 1, facecolor="gold", alpha=0.8
            )
            self.ax.add_patch(rect)

        # Taxi (mavi)
        rect = patches.Rectangle(
            (self.taxi_col, self.taxi_row), 1, 1, facecolor="lightskyblue", alpha=0.9
        )
        self.ax.add_patch(rect)

        # Duvarlar (kalın siyah)
        # Duvarlar (kalın siyah)
        for (cell1, cell2) in self.walls:
            (r1, c1), (r2, c2) = cell1, cell2

            # Aynı satırdaysa (r1 == r2): hücreler YATAY komşu (sol-sağ)
            # Aradaki sınır, dikey bir çizgidir: x = max(c1, c2)
            if r1 == r2:
                x = max(c1, c2)  # iki hücre arasındaki dikey kenar
                y1 = r1
                y2 = r1 + 1
                self.ax.plot([x, x], [y1, y2], linewidth=3, color="black")

            # Aynı sütundaysa: hücreler DİKEY komşu (üst-alt)
            # Aradaki sınır, yatay bir çizgidir: y = max(r1, r2)
            else:
                y = max(r1, r2)  # iki hücre arasındaki yatay kenar
                x1 = c1
                x2 = c1 + 1
                self.ax.plot([x1, x2], [y, y], linewidth=3, color="black")

        self.ax.set_title(
            f"Episode {self.render_episode_idx} | Step {self.render_step_idx}",
            fontsize=10,
        )

        self.fig.tight_layout()
        self.fig.canvas.draw()

        if mode == "human":
            plt.show(block=False)
            return None

        buf = self.fig.canvas.buffer_rgba()
        w, h = self.fig.canvas.get_width_height()
        image = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)  # RGBA
        image = image[:, :, :3].copy()
        return image
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None

def train_q_learning(
    episodes=80000,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.09,
):
    env = TaxiEnv6x6()
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    epsilon = epsilon_start

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if ep % 1000 == 0:
            print(
                f"Episode {ep}/{episodes} | Epsilon={epsilon:.3f} | "
                f"Total reward={total_reward:.1f}"
            )

    env.close()
    np.save(Q_FILE, Q)
    print(f"\nQ tablosu '{Q_FILE}' dosyasına kaydedildi.")
    return Q

def watch_trained_agent(num_episodes=1, q_table=None, pause_time=0.3, epsilon_eval=0.05):

    if q_table is None:
        q_table = np.load(Q_FILE)

    env = TaxiEnv6x6(render_mode="human")

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False

        env.render_episode_idx = ep
        env.render_step_idx = 0
        env.render(mode="human")
        plt.pause(pause_time)

        while not done:
            if random.random() < epsilon_eval:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            env.render_step_idx += 1
            env.render(mode="human")
            plt.pause(pause_time)

            state = next_state

    env.close()

def save_run_as_gif(q_table=None,
                    max_steps=60,
                    gif_name="taxi_episode.gif",
                    frame_duration_ms=250):
    """
    Q-tablosuna göre TAMAMEN greedy (argmax) oynayan taksinin
    bir bölümünü GIF olarak kaydeder. Rastgelelik YOK.
    """
    from PIL import Image

    if q_table is None:
        q_table = np.load(Q_FILE)

    env = TaxiEnv6x6(render_mode="rgb_array")

    frames = []
    state, _ = env.reset()
    done = False
    step = 0

    env.render_episode_idx = 1
    env.render_step_idx = 0
    frames.append(env.render(mode="rgb_array").copy())

    while not done and step < max_steps:
        # >>> HİÇ RANDOM YOK: tam greedy politika <<<
        action = int(np.argmax(q_table[state]))

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        step += 1
        env.render_step_idx = step
        frames.append(env.render(mode="rgb_array").copy())

        state = next_state

    env.close()

    print("GIF için kare sayısı:", len(frames))

    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        gif_name,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=frame_duration_ms,
    )

    print(f"Animasyonlu GIF '{gif_name}' olarak kaydedildi.")



# ============================================================
#                      MAIN BLOKU
# ============================================================

if __name__ == "__main__":
    DO_TRAIN = True          # Q tablon hazırsa False bırak

    if DO_TRAIN:
        Q = train_q_learning()
    else:
        Q = np.load(Q_FILE)
        print("Q tablosu yüklendi.")

    # İstersen önce canlı bak:
    watch_trained_agent(num_episodes=3, q_table=Q, pause_time=0.3)

    # Sonra aynı politikayı GIF olarak kaydet:
    save_run_as_gif(q_table=Q, max_steps=60, gif_name="taxi_episode.gif")

