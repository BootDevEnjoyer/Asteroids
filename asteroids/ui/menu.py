"""Menu screen and UI components for game state selection."""

import pygame
from typing import Optional, List, Tuple
from asteroids.core.constants import SCREEN_WIDTH, SCREEN_HEIGHT
from asteroids.ui.starfield import Starfield


class Button:
    """Clickable button with hover effects."""

    # Class-level font cache (shared across all buttons)
    _font: Optional[pygame.font.Font] = None
    _subtitle_font: Optional[pygame.font.Font] = None

    @classmethod
    def _get_fonts(cls) -> Tuple[pygame.font.Font, pygame.font.Font]:
        """Lazily initialize and return cached fonts."""
        if cls._font is None:
            cls._font = pygame.font.Font(None, 36)
            cls._subtitle_font = pygame.font.Font(None, 24)
        return cls._font, cls._subtitle_font

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        subtitle: str = "",
        color: Tuple[int, int, int] = (40, 40, 80),
        hover_color: Tuple[int, int, int] = (60, 60, 120),
        border_color: Tuple[int, int, int] = (100, 150, 255),
    ):
        self.rect = rect
        self.text = text
        self.subtitle = subtitle
        self.color = color
        self.hover_color = hover_color
        self.border_color = border_color
        self.is_hovered = False

    def update(self, mouse_pos: Tuple[int, int]) -> None:
        """Update hover state based on mouse position."""
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def draw(self, screen: pygame.Surface) -> None:
        """Render the button with current state."""
        current_color = self.hover_color if self.is_hovered else self.color
        border_width = 3 if self.is_hovered else 2

        # Button background
        pygame.draw.rect(screen, current_color, self.rect, border_radius=8)
        pygame.draw.rect(
            screen, self.border_color, self.rect, border_width, border_radius=8
        )

        # Main text
        font, subtitle_font = self._get_fonts()
        text_surface = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect()
        text_rect.centerx = self.rect.centerx
        text_rect.centery = self.rect.centery - (10 if self.subtitle else 0)
        screen.blit(text_surface, text_rect)

        # Subtitle text
        if self.subtitle:
            subtitle_color = (180, 180, 200) if not self.is_hovered else (220, 220, 240)
            subtitle_surface = subtitle_font.render(self.subtitle, True, subtitle_color)
            subtitle_rect = subtitle_surface.get_rect()
            subtitle_rect.centerx = self.rect.centerx
            subtitle_rect.centery = self.rect.centery + 18
            screen.blit(subtitle_surface, subtitle_rect)

    def is_clicked(self, mouse_pos: Tuple[int, int], mouse_pressed: bool) -> bool:
        """Check if button was clicked."""
        return mouse_pressed and self.rect.collidepoint(mouse_pos)


class MenuScreen:
    """Main menu screen with game mode selection buttons."""

    def __init__(self):
        self.starfield = Starfield(num_layers=3, stars_per_layer=100)
        self.buttons: List[Tuple[Button, str]] = []
        self._create_buttons()
        
        # Cache fonts for title and footer rendering
        self._title_font = pygame.font.Font(None, 72)
        self._subtitle_font = pygame.font.Font(None, 32)
        self._footer_font = pygame.font.Font(None, 24)

    def _create_buttons(self) -> None:
        """Initialize menu buttons with their target states."""
        button_width = 400
        button_height = 70
        button_x = (SCREEN_WIDTH - button_width) // 2
        spacing = 90
        start_y = SCREEN_HEIGHT // 2 - 40

        # Button definitions: (text, subtitle, target_state, border_color)
        button_defs = [
            (
                "Watch AI Learn",
                "See neural networks train in real-time",
                "RL_TRAINING",
                (100, 255, 100),
            ),
            (
                "Watch Trained AI",
                "Observe the AI's learned behaviors",
                "RL_SHOWCASE",
                (255, 200, 100),
            ),
            (
                "Play Classic Asteroids",
                "Manual gameplay with traditional enemies",
                "CLASSIC_PLAY",
                (100, 150, 255),
            ),
            (
                "Reset AI Model",
                "Archive current model and start fresh training",
                "RESET_AI",
                (255, 100, 100),
            ),
        ]

        for i, (text, subtitle, state, border_color) in enumerate(button_defs):
            rect = pygame.Rect(button_x, start_y + i * spacing, button_width, button_height)
            button = Button(
                rect=rect,
                text=text,
                subtitle=subtitle,
                border_color=border_color,
            )
            self.buttons.append((button, state))

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """
        Process a pygame event and return target state name if button clicked.
        
        Returns:
            State name string (e.g., "RL_TRAINING") or None if no action.
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            for button, state_name in self.buttons:
                if button.is_clicked(mouse_pos, True):
                    return state_name
        return None

    def update(self, dt: float) -> None:
        """Update menu animations and button states."""
        self.starfield.update(dt)
        mouse_pos = pygame.mouse.get_pos()
        for button, _ in self.buttons:
            button.update(mouse_pos)

    def draw(self, screen: pygame.Surface) -> None:
        """Render the complete menu screen."""
        # Background
        screen.fill((5, 5, 15))
        self.starfield.draw(screen)
        self.starfield.add_twinkle_effect(screen)

        # Title
        self._draw_title(screen)

        # Buttons
        for button, _ in self.buttons:
            button.draw(screen)

        # Footer
        self._draw_footer(screen)

    def _draw_title(self, screen: pygame.Surface) -> None:
        """Render the game title and subtitle."""
        # Main title
        title_text = self._title_font.render("ASTEROIDS", True, (255, 255, 255))
        title_rect = title_text.get_rect()
        title_rect.centerx = SCREEN_WIDTH // 2
        title_rect.y = 80
        screen.blit(title_text, title_rect)

        # Subtitle
        subtitle_text = self._subtitle_font.render(
            "Neural Network AI Demonstration", True, (150, 180, 255)
        )
        subtitle_rect = subtitle_text.get_rect()
        subtitle_rect.centerx = SCREEN_WIDTH // 2
        subtitle_rect.y = 145
        screen.blit(subtitle_text, subtitle_rect)

        # Decorative line
        line_y = 185
        line_width = 300
        line_start = (SCREEN_WIDTH // 2 - line_width // 2, line_y)
        line_end = (SCREEN_WIDTH // 2 + line_width // 2, line_y)
        pygame.draw.line(screen, (80, 100, 150), line_start, line_end, 2)

    def _draw_footer(self, screen: pygame.Surface) -> None:
        """Render footer instructions."""
        footer_text = self._footer_font.render("Press ESC to exit", True, (100, 100, 120))
        footer_rect = footer_text.get_rect()
        footer_rect.centerx = SCREEN_WIDTH // 2
        footer_rect.y = SCREEN_HEIGHT - 50
        screen.blit(footer_text, footer_rect)

