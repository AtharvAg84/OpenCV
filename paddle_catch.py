import cv2
import numpy as np

# Set up the game window dimensions
WIDTH, HEIGHT = 800, 600
PADDLE_WIDTH, PADDLE_HEIGHT = 100, 20
BLOCK_SIZE = 30
BLOCK_SPEED = 5
PADDLE_SPEED = 30

# Initialize variables for paddle position and block position
paddle_x = (WIDTH - PADDLE_WIDTH) // 2
block_x = np.random.randint(0, WIDTH - BLOCK_SIZE)
block_y = 0
score = 0
game_over = False

# Create an OpenCV window
cv2.namedWindow("Catch the Block")

# Game loop
while True:
    # Create a blank image to represent the game screen
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Check for key press events
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # Escape key to quit
        break

    if key == ord('a') and paddle_x > 0:  # Left arrow (move left)
        paddle_x -= PADDLE_SPEED
    elif key == ord('d') and paddle_x < WIDTH - PADDLE_WIDTH:  # Right arrow (move right)
        paddle_x += PADDLE_SPEED

    # Update the block position
    block_y += BLOCK_SPEED

    # If the block goes off the screen, reset it
    if block_y > HEIGHT:
        block_y = 0
        block_x = np.random.randint(0, WIDTH - BLOCK_SIZE)
        
        # If the block isn't caught, end the game
        if paddle_x > block_x + BLOCK_SIZE or paddle_x + PADDLE_WIDTH < block_x:
            game_over = True
            break

    # Check if the block is caught by the paddle
    if block_y + BLOCK_SIZE >= HEIGHT - PADDLE_HEIGHT and paddle_x < block_x + BLOCK_SIZE and paddle_x + PADDLE_WIDTH > block_x:
        score += 1
        block_y = 0
        block_x = np.random.randint(0, WIDTH - BLOCK_SIZE)

    # Draw the falling block (rectangle)
    cv2.rectangle(frame, (block_x, block_y), (block_x + BLOCK_SIZE, block_y + BLOCK_SIZE), (0, 0, 255), -1)

    # Draw the paddle (rectangle)
    cv2.rectangle(frame, (paddle_x, HEIGHT - PADDLE_HEIGHT), (paddle_x + PADDLE_WIDTH, HEIGHT), (0, 255, 0), -1)

    # Display the score on the screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Score: {score}", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the game window
    cv2.imshow("Catch the Block", frame)

    # If game over, display a message and exit
    if game_over:
        cv2.putText(frame, "Game Over! Press ESC to exit", (250, HEIGHT // 2), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Catch the Block", frame)
        cv2.waitKey(0)  # Wait for the user to press a key
        break

# Close the game window
cv2.destroyAllWindows()
