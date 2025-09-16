# import pygame and initialize
import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
BOID_NUM = 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up clock for 60 FPS
clock = pygame.time.Clock()

GRID_SIZE = 40  # size of each grid cell

def random_speed():
    return random.uniform(-2, 2), random.uniform(-2, 2)

def get_grid_pos(x, y):
    return int(x // GRID_SIZE), int(y // GRID_SIZE)

def build_grid(boids, grid, grid_cells):
    for cell in grid_cells:
        grid[cell] = []
    for b in boids:
        gx, gy = get_grid_pos(b.x, b.y)
        grid[(gx, gy)].append(b)

class Boid:
    def __init__(self, x=None, y=None, radius=3, color=(173, 216, 230)): 
        self.x = x if x is not None else random.uniform(20, WIDTH-20)
        self.y = y if y is not None else random.uniform(20, HEIGHT-20)
        self.radius = radius
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 2)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.ax = 0
        self.ay = 0
        self.ali_weight_var = random.uniform(0.9, 1.1)
        self.coh_weight_var = random.uniform(0.9, 1.1)
    
    def wrap_delta(self, dx, dy):
        # Compute shortest vector considering wrapping
        if dx > WIDTH / 2:
            dx -= WIDTH
        elif dx < -WIDTH / 2:
            dx += WIDTH
        if dy > HEIGHT / 2:
            dy -= HEIGHT
        elif dy < -HEIGHT / 2:
            dy += HEIGHT
        return dx, dy

    def in_view(self, other, fov=240):  # Default FOV set to 240 (between 220 and 260)
        dx = other.x - self.x
        dy = other.y - self.y
        dx, dy = self.wrap_delta(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))
        heading = math.degrees(math.atan2(self.vy, self.vx))
        diff = (angle - heading + 360) % 360
        return diff < fov / 2 or diff > 360 - fov / 2

    def move(self):
        self.x += self.vx
        self.y += self.vy
        # Edge wrapping for seamless movement
        if self.x < 0:
            self.x += WIDTH
        elif self.x > WIDTH:
            self.x -= WIDTH
        if self.y < 0:
            self.y += HEIGHT
        elif self.y > HEIGHT:
            self.y -= HEIGHT

    def get_neighbors(self, grid, neighbor_dist):
        gx, gy = get_grid_pos(self.x, self.y)
        cells = [
            (gx + dx, gy + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        ]
        neighbors = []
        for cell in cells:
            for other in grid.get(cell, []):
                if other is not self:
                    dx = other.x - self.x
                    dy = other.y - self.y
                    dx, dy = self.wrap_delta(dx, dy)
                    dist = math.hypot(dx, dy)
                    if dist < neighbor_dist:
                        neighbors.append((other, dx, dy, dist))
        return neighbors

    def update(self, grid, grid_cells, neighbor_buf):
        # --- Parameters ---
        max_speed = 3.2
        min_speed = 1.2
        max_force = 0.045  # slightly higher for higher weights
        sep_dist = 22
        ali_dist = 70      # 0.7 of coh_dist
        coh_dist = 100
        sep_dist2 = sep_dist * sep_dist
        ali_dist2 = ali_dist * ali_dist
        coh_dist2 = coh_dist * coh_dist
        min_dist2 = 8 * 8
        ali_fov_angle = 200  # alignment FOV
        ali_fov_cos = math.cos(math.radians(ali_fov_angle / 2))

        # --- Neighbor search ---
        gx, gy = get_grid_pos(self.x, self.y)
        neighbor_buf.clear()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (gx + dx, gy + dy)
                for other in grid.get(cell, []):
                    if other is self:
                        continue
                    dx_ = other.x - self.x
                    dy_ = other.y - self.y
                    # Wrap deltas inline
                    if dx_ > WIDTH / 2:
                        dx_ -= WIDTH
                    elif dx_ < -WIDTH / 2:
                        dx_ += WIDTH
                    if dy_ > HEIGHT / 2:
                        dy_ -= HEIGHT
                    elif dy_ < -HEIGHT / 2:
                        dy_ += HEIGHT
                    dist2 = dx_ * dx_ + dy_ * dy_
                    neighbor_buf.append((other, dx_, dy_, dist2))

        # --- Steering accumulators ---
        sep_x = sep_y = sep_n = 0
        ali_x = ali_y = ali_n = 0
        coh_x = coh_y = coh_n = 0

        # --- Precompute heading for dot product FOV ---
        vmag = math.hypot(self.vx, self.vy)
        if vmag > 1e-8:
            vx_norm = self.vx / vmag
            vy_norm = self.vy / vmag
        else:
            vx_norm = 1.0
            vy_norm = 0.0

        for other, dx_, dy_, dist2 in neighbor_buf:
            # Dot product FOV test for alignment only
            if dist2 > 1e-8:
                dot = (dx_ * vx_norm + dy_ * vy_norm) / (math.sqrt(dist2))
            else:
                dot = 1.0
            ali_in_fov = (dot > ali_fov_cos)
            # Separation: always include if very close, else FOV (use ali_in_fov for simplicity)
            if dist2 < sep_dist2 and (ali_in_fov or dist2 < min_dist2):
                if dist2 > 1e-8:
                    inv_dist = 1.0 / math.sqrt(dist2)
                    sep_x -= dx_ * inv_dist
                    sep_y -= dy_ * inv_dist
                    sep_n += 1
            # Alignment: FOV or very close
            if dist2 < ali_dist2 and (ali_in_fov or dist2 < min_dist2):
                ali_x += other.vx
                ali_y += other.vy
                ali_n += 1
            # Cohesion: ignore FOV, just distance
            if dist2 < coh_dist2:
                coh_x += self.x + dx_
                coh_y += self.y + dy_
                coh_n += 1

        # --- Compute steering vectors ---
        steer_sep_x = steer_sep_y = 0
        if sep_n:
            sep_x /= sep_n
            sep_y /= sep_n
            mag = math.hypot(sep_x, sep_y)
            if mag > 1e-8:
                steer_sep_x = (sep_x / mag) * max_speed - self.vx
                steer_sep_y = (sep_y / mag) * max_speed - self.vy

        steer_ali_x = steer_ali_y = 0
        if ali_n:
            ali_x /= ali_n
            ali_y /= ali_n
            mag = math.hypot(ali_x, ali_y)
            if mag > 1e-8:
                steer_ali_x = (ali_x / mag) * max_speed - self.vx
                steer_ali_y = (ali_y / mag) * max_speed - self.vy

        steer_coh_x = steer_coh_y = 0
        if coh_n:
            center_x = coh_x / coh_n
            center_y = coh_y / coh_n
            dx = center_x - self.x
            dy = center_y - self.y
            mag = math.hypot(dx, dy)
            if mag > 1e-8:
                steer_coh_x = (dx / mag) * max_speed - self.vx
                steer_coh_y = (dy / mag) * max_speed - self.vy

        # --- Compose and clamp ---
        jitter_x = random.uniform(-0.05, 0.05)
        jitter_y = random.uniform(-0.05, 0.05)
        steer_x = (
            0.22 * steer_sep_x +
            0.06 * steer_ali_x * self.ali_weight_var +
            0.16 * steer_coh_x * self.coh_weight_var +
            jitter_x
        )
        steer_y = (
            0.22 * steer_sep_y +
            0.06 * steer_ali_y * self.ali_weight_var +
            0.16 * steer_coh_y * self.coh_weight_var +
            jitter_y
        )
        steer_mag = math.hypot(steer_x, steer_y)
        if steer_mag > max_force:
            steer_x = (steer_x / steer_mag) * max_force
            steer_y = (steer_y / steer_mag) * max_force

        self.ax = steer_x
        self.ay = steer_y
        self.vx += self.ax
        self.vy += self.ay
        speed = math.hypot(self.vx, self.vy)
        if speed > max_speed:
            self.vx = (self.vx / speed) * max_speed
            self.vy = (self.vy / speed) * max_speed
        elif speed < min_speed:
            self.vx = (self.vx / speed) * min_speed
            self.vy = (self.vy / speed) * min_speed

    def draw(self, surface):
        # Draw a triangle pointing in the direction of velocity
        angle = math.atan2(self.vy, self.vx)
        p1 = (self.x + math.cos(angle) * (self.radius * 2.5),
              self.y + math.sin(angle) * (self.radius * 2.5))
        p2 = (self.x + math.cos(angle + 2.5) * self.radius,
              self.y + math.sin(angle + 2.5) * self.radius)
        p3 = (self.x + math.cos(angle - 2.5) * self.radius,
              self.y + math.sin(angle - 2.5) * self.radius)
        pygame.draw.polygon(surface, self.color, [p1, p2, p3])
        
    def distance(self, other):
        dx = other.x - self.x
        dy = other.y - self.y
        dx, dy = self.wrap_delta(dx, dy)
        return math.hypot(dx, dy)

    def separation(self, grid, separation_dist=25, min_dist=8):
        steer_x, steer_y, count = 0, 0, 0
        for other, dx, dy, dist in self.get_neighbors(grid, separation_dist):
            if (self.in_view(other) or dist < min_dist):
                if dist > 0:
                    steer_x -= dx / dist
                    steer_y -= dy / dist
                    count += 1
        if count:
            steer_x /= count
            steer_y /= count
            mag = math.hypot(steer_x, steer_y)
            if mag > 0:
                steer_x = (steer_x / mag) * 3.2 - self.vx
                steer_y = (steer_y / mag) * 3.2 - self.vy
            else:
                steer_x, steer_y = 0, 0
        return steer_x, steer_y

    def alignment(self, grid, neighbor_dist=50, min_dist=8):
        avg_vx, avg_vy, count = 0, 0, 0
        for other, dx, dy, dist in self.get_neighbors(grid, neighbor_dist):
            if (self.in_view(other) or dist < min_dist):
                avg_vx += other.vx
                avg_vy += other.vy
                count += 1
        if count:
            avg_vx /= count
            avg_vy /= count
            mag = math.hypot(avg_vx, avg_vy)
            if mag > 0:
                steer_x = (avg_vx / mag) * 3.2 - self.vx
                steer_y = (avg_vy / mag) * 3.2 - self.vy
            else:
                steer_x, steer_y = 0, 0
            return steer_x, steer_y
        return 0, 0

    def cohesion(self, grid, neighbor_dist=50, min_dist=8):
        center_x, center_y, count = 0, 0, 0
        for other, dx, dy, dist in self.get_neighbors(grid, neighbor_dist):
            # Ignore FOV for cohesion, always include if in distance
            if dist < neighbor_dist or dist < min_dist:
                center_x += self.x + dx
                center_y += self.y + dy
                count += 1
        if count:
            center_x /= count
            center_y /= count
            desired_x = center_x - self.x
            desired_y = center_y - self.y
            mag = math.hypot(desired_x, desired_y)
            if mag > 0:
                steer_x = (desired_x / mag) * 3.2 - self.vx
                steer_y = (desired_y / mag) * 3.2 - self.vy
            else:
                steer_x, steer_y = 0, 0
            return steer_x, steer_y
        return 0, 0

boids = [Boid() for _ in range(BOID_NUM)]
grid_cells = [(gx, gy) for gx in range((WIDTH // GRID_SIZE) + 2) for gy in range((HEIGHT // GRID_SIZE) + 2)]
grid = {}
neighbor_buf = []

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    build_grid(boids, grid, grid_cells)
    for b in boids:
        b.update(grid, grid_cells, neighbor_buf)
        b.move()
        b.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

