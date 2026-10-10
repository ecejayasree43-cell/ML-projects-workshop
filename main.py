import tkinter as tk
from tkinter import ttk, messagebox
from mood_predictor import predict_mood
import pandas as pd
import pygame
import os
from PIL import Image, ImageTk


# =========================
# PATHS
# =========================

BACKGROUND_PATH = "assets/melora_bgim.png"
LOGO_PATH = "assets/melora_logo.png"
SONGS_PATH = "data/songs.csv"
MUSIC_FOLDER = "music"


# =========================
# COLORS
# =========================

BG_DARK = "#100C1C"
PANEL = "#1B152A"
PANEL_LIGHT = "#251D38"

WHITE = "#FFFFFF"
LIGHT_TEXT = "#E9E2F5"
MUTED_TEXT = "#B8ADC9"

PURPLE = "#9B5CFF"
PINK = "#FF4FA3"
BLUE = "#4D9EFF"
CYAN = "#42D9FF"
YELLOW = "#FFD166"
GREEN = "#55E6A5"
RED = "#FF5C7A"


# =========================
# LOAD SONG DATA
# =========================

songs = pd.read_csv(SONGS_PATH)

recommended_songs = []

current_song_index = -1
current_file = None
song_duration = 0
is_paused = False


# =========================
# HELPER
# =========================

def format_time(seconds):

    seconds = int(seconds)

    minutes = seconds // 60
    seconds = seconds % 60

    return f"{minutes}:{seconds:02d}"


# =========================
# RECOMMEND SONGS
# =========================


def show_recommendations(selected_mood=None, confidence=None):

    global recommended_songs
    global current_song_index

    # Use the dropdown when no mood is passed by the ML model
    if selected_mood is None:
        selected_mood = mood_combo.get()

    if not selected_mood:
        messagebox.showwarning(
            "Select Mood",
            "Please select or describe your mood first!"
        )
        return

    # Update dropdown to reflect the predicted mood
    mood_combo.set(selected_mood)

    recommended_songs = songs[
        songs["mood"].astype(str).str.strip().str.lower()
        == selected_mood.strip().lower()
    ]

    song_listbox.delete(0, tk.END)

    for _, song in recommended_songs.iterrows():
        display_text = (
            f"{song['song_title']}   • {song['movie']}"
        )
        song_listbox.insert(tk.END, display_text)

    current_song_index = -1

    if confidence is not None:
        mood_display.config(
            text=f"AI Mood: {selected_mood} ({confidence:.1f}%)"
        )
    else:
        mood_display.config(
            text=f"Mood: {selected_mood}"
        )

    if recommended_songs.empty:
        now_playing_label.config(
            text=f"No songs found for {selected_mood}"
        )
    else:
        now_playing_label.config(
            text=f"{len(recommended_songs)} songs found"
        )

# =========================
# PLAY SONG
# =========================

def play_selected_song():

    global current_song_index
    global current_file
    global song_duration
    global is_paused

    selected = song_listbox.curselection()

    if not selected:

        messagebox.showwarning(
            "Select Song",
            "Please select a song from the list."
        )

        return

    current_song_index = selected[0]

    song = recommended_songs.iloc[current_song_index]

    file_name = str(song["file_name"]).strip()

    file_path = os.path.join(
        MUSIC_FOLDER,
        file_name
    )

    if not os.path.exists(file_path):

        messagebox.showerror(
            "Audio Not Found",
            f"Audio file not found:\n\n{file_name}"
        )

        return

    try:

        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        current_file = file_path
        is_paused = False

        # Get actual song duration
        sound = pygame.mixer.Sound(file_path)
        song_duration = sound.get_length()

        progress_bar["maximum"] = song_duration
        progress_bar["value"] = 0

        current_time_label.config(
            text="0:00"
        )

        total_time_label.config(
            text=format_time(song_duration)
        )

        now_playing_label.config(
            text=f"🎵 {song['song_title']}"
        )

        song_info_label.config(
            text=(
                f"🎬 Movie: {song['movie']}\n"
                f"🎼 Composer: {song['composer']}\n"
                f"📅 Year: {song['year']}\n"
                f"⚡ Energy: {song['energy_level']}\n"
                f"💜 Mood: {song['mood']}"
            )
        )

        update_progress()

    except Exception as e:

        messagebox.showerror(
            "Playback Error",
            str(e)
        )


# =========================
# PROGRESS BAR
# =========================

def update_progress():

    if pygame.mixer.music.get_busy() and not is_paused:

        current_position = pygame.mixer.music.get_pos() / 1000

        progress_bar["value"] = current_position

        current_time_label.config(
            text=format_time(current_position)
        )

    root.after(500, update_progress)


# =========================
# PAUSE
# =========================

def pause_song():

    global is_paused

    if pygame.mixer.music.get_busy():

        pygame.mixer.music.pause()

        is_paused = True

        now_playing_label.config(
            text="⏸ Song Paused"
        )


# =========================
# RESUME
# =========================

def resume_song():

    global is_paused

    pygame.mixer.music.unpause()

    is_paused = False

    now_playing_label.config(
        text="▶ Song Resumed"
    )


# =========================
# STOP
# =========================

def stop_song():

    global is_paused

    pygame.mixer.music.stop()

    is_paused = False

    progress_bar["value"] = 0

    current_time_label.config(
        text="0:00"
    )

    now_playing_label.config(
        text="⏹ Song Stopped"
    )


# =========================
# VOLUME
# =========================

def change_volume(value):

    volume = float(value) / 100

    pygame.mixer.music.set_volume(volume)


# =========================
# CLOSE APP
# =========================

def close_app():

    pygame.mixer.music.stop()

    pygame.quit()

    root.destroy()


# =========================
# TKINTER WINDOW
# =========================

root = tk.Tk()

root.title("🎵 AI MoodTuner")

root.geometry("1100x780")

root.resizable(False, False)

root.protocol(
    "WM_DELETE_WINDOW",
    close_app
)


# =========================
# BACKGROUND
# =========================

try:

    bg_image = Image.open(BACKGROUND_PATH)

    bg_image = bg_image.resize(
        (1100, 780)
    )

    bg_photo = ImageTk.PhotoImage(bg_image)

    background_label = tk.Label(
        root,
        image=bg_photo
    )

    background_label.place(
        x=0,
        y=0,
        relwidth=1,
        relheight=1
    )

except Exception:

    root.configure(
        bg=BG_DARK
    )


# =========================
# MAIN CONTAINER
# =========================

main_frame = tk.Frame(
    root,
    bg=BG_DARK
)

main_frame.place(
    x=30,
    y=25,
    width=1040,
    height=730
)


# =========================
# HEADER
# =========================

header = tk.Frame(
    main_frame,
    bg=BG_DARK
)

header.pack(
    fill="x",
    padx=25,
    pady=15
)


# LOGO

try:

    logo_image = Image.open(LOGO_PATH)

    logo_image.thumbnail(
        (90, 90)
    )

    logo_photo = ImageTk.PhotoImage(
        logo_image
    )

    logo_label = tk.Label(
        header,
        image=logo_photo,
        bg=BG_DARK
    )

    logo_label.pack(
        side="left",
        padx=(0, 18)
    )

except Exception:

    pass


title_frame = tk.Frame(
    header,
    bg=BG_DARK
)

title_frame.pack(
    side="left"
)


tk.Label(
    title_frame,
    text="🎵 AI MOODTUNER",
    font=("Segoe UI", 28, "bold"),
    fg=WHITE,
    bg=BG_DARK
).pack(
    anchor="w"
)


tk.Label(
    title_frame,
    text="✨ Your Mood • Your Music • Your Moment",
    font=("Segoe UI", 12),
    fg=PINK,
    bg=BG_DARK
).pack(
    anchor="w"
)


tk.Label(
    title_frame,
    text="Discover Tamil songs that match your mood",
    font=("Segoe UI", 10),
    fg=MUTED_TEXT,
    bg=BG_DARK
).pack(
    anchor="w"
)


# =========================
# MOOD SECTION
# =========================

mood_card = tk.Frame(
    main_frame,
    bg=PANEL
)

mood_card.pack(
    fill="x",
    padx=25,
    pady=10
)


tk.Label(
    mood_card,
    text="💫 HOW ARE YOU FEELING TODAY?",
    font=("Segoe UI", 13, "bold"),
    fg=WHITE,
    bg=PANEL
).pack(
    side="left",
    padx=20,
    pady=15
)


mood_combo = ttk.Combobox(
    mood_card,
    values=sorted(
        songs["mood"].dropna().unique()
    ),
    state="readonly",
    width=22,
    font=("Segoe UI", 11)
)

mood_combo.pack(
    side="left",
    padx=10
)


recommend_button = tk.Button(
    mood_card,
    text="✨ RECOMMEND MUSIC",
    command=show_recommendations,
    bg=PURPLE,
    fg=WHITE,
    font=("Segoe UI", 10, "bold"),
    relief="flat",
    padx=18,
    pady=8,
    cursor="hand2"
)

recommend_button.pack(
    side="left",
    padx=10
)


mood_display = tk.Label(
    mood_card,
    text="💜 Mood: —",
    font=("Segoe UI", 10, "bold"),
    fg=YELLOW,
    bg=PANEL
)

mood_display.pack(
    side="right",
    padx=20
)



# =========================
# AI TEXT MOOD INPUT
# =========================

text_mood_card = tk.Frame(
    main_frame,
    bg=PANEL
)

text_mood_card.pack(
    fill="x",
    padx=25,
    pady=5
)

tk.Label(
    text_mood_card,
    text="DESCRIBE YOUR FEELING",
    font=("Segoe UI", 11, "bold"),
    fg=WHITE,
    bg=PANEL
).pack(
    side="left",
    padx=(15, 10),
    pady=12
)

mood_text_entry = tk.Entry(
    text_mood_card,
    font=("Segoe UI", 11),
    bg=PANEL_LIGHT,
    fg=WHITE,
    insertbackground=WHITE,
    relief="flat",
    width=38
)

mood_text_entry.pack(
    side="left",
    padx=8,
    ipady=7
)


def analyze_text_mood(event=None):

    user_text = mood_text_entry.get().strip()

    if not user_text:
        messagebox.showwarning(
            "Input Required",
            "Please describe how you are feeling."
        )
        return

    try:
        predicted_mood, confidence = predict_mood(user_text)

        if predicted_mood is None:
            messagebox.showwarning(
                "Input Required",
                "Please enter a valid mood description."
            )
            return

        # Predict mood, then recommend matching songs
        show_recommendations(
            selected_mood=predicted_mood,
            confidence=confidence
        )

    except Exception as error:
        messagebox.showerror(
            "Mood Prediction Error",
            str(error)
        )


tk.Button(
    text_mood_card,
    text="ANALYZE WITH AI",
    command=analyze_text_mood,
    bg=PINK,
    fg=WHITE,
    font=("Segoe UI", 9, "bold"),
    relief="flat",
    padx=12,
    pady=8,
    cursor="hand2"
).pack(
    side="left",
    padx=8
)

# Press Enter to analyze the text
mood_text_entry.bind(
    "<Return>",
    analyze_text_mood
)


# =========================
# CONTENT AREA
# =========================

content_frame = tk.Frame(
    main_frame,
    bg=BG_DARK
)

content_frame.pack(
    fill="both",
    expand=True,
    padx=25,
    pady=10
)


# =========================
# LEFT SONG LIST
# =========================

songs_card = tk.Frame(
    content_frame,
    bg=PANEL
)

songs_card.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 10)
)


tk.Label(
    songs_card,
    text="🎶 RECOMMENDED SONGS",
    font=("Segoe UI", 14, "bold"),
    fg=WHITE,
    bg=PANEL
).pack(
    anchor="w",
    padx=20,
    pady=(18, 10)
)


song_listbox = tk.Listbox(
    songs_card,
    bg=PANEL_LIGHT,
    fg=LIGHT_TEXT,
    selectbackground=PURPLE,
    selectforeground=WHITE,
    font=("Segoe UI", 10),
    relief="flat",
    highlightthickness=0
)

song_listbox.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=(0, 15)
)


# =========================
# RIGHT PLAYER
# =========================

player_card = tk.Frame(
    content_frame,
    bg=PANEL
)

player_card.pack(
    side="right",
    fill="both",
    expand=True,
    padx=(10, 0)
)


tk.Label(
    player_card,
    text="🎧 NOW PLAYING",
    font=("Segoe UI", 14, "bold"),
    fg=WHITE,
    bg=PANEL
).pack(
    pady=(18, 5)
)


tk.Label(
    player_card,
    text="♫",
    font=("Segoe UI", 55, "bold"),
    fg=PINK,
    bg=PANEL
).pack(
    pady=5
)


now_playing_label = tk.Label(
    player_card,
    text="Select a song to start",
    font=("Segoe UI", 12, "bold"),
    fg=CYAN,
    bg=PANEL
)

now_playing_label.pack(
    pady=5
)


# =========================
# SONG DETAILS
# =========================

song_info_label = tk.Label(
    player_card,
    text="🎵 Song information will appear here",
    font=("Segoe UI", 9),
    fg=MUTED_TEXT,
    bg=PANEL,
    justify="left"
)

song_info_label.pack(
    pady=10
)


# =========================
# PROGRESS BAR
# =========================

progress_frame = tk.Frame(
    player_card,
    bg=PANEL
)

progress_frame.pack(
    fill="x",
    padx=25,
    pady=10
)


current_time_label = tk.Label(
    progress_frame,
    text="0:00",
    font=("Segoe UI", 9),
    fg=WHITE,
    bg=PANEL
)

current_time_label.pack(
    side="left"
)


progress_bar = ttk.Progressbar(
    progress_frame,
    orient="horizontal",
    mode="determinate",
    length=250
)

progress_bar.pack(
    side="left",
    fill="x",
    expand=True,
    padx=10
)


total_time_label = tk.Label(
    progress_frame,
    text="0:00",
    font=("Segoe UI", 9),
    fg=WHITE,
    bg=PANEL
)

total_time_label.pack(
    side="right"
)


# =========================
# MUSIC CONTROLS
# =========================

controls = tk.Frame(
    player_card,
    bg=PANEL
)

controls.pack(
    pady=15
)


tk.Button(
    controls,
    text="▶ PLAY",
    command=play_selected_song,
    bg=GREEN,
    fg=BG_DARK,
    font=("Segoe UI", 9, "bold"),
    relief="flat",
    padx=12,
    pady=7,
    cursor="hand2"
).grid(
    row=0,
    column=0,
    padx=4
)


tk.Button(
    controls,
    text="⏸ PAUSE",
    command=pause_song,
    bg=YELLOW,
    fg=BG_DARK,
    font=("Segoe UI", 9, "bold"),
    relief="flat",
    padx=12,
    pady=7,
    cursor="hand2"
).grid(
    row=0,
    column=1,
    padx=4
)


tk.Button(
    controls,
    text="▶ RESUME",
    command=resume_song,
    bg=CYAN,
    fg=BG_DARK,
    font=("Segoe UI", 9, "bold"),
    relief="flat",
    padx=12,
    pady=7,
    cursor="hand2"
).grid(
    row=0,
    column=2,
    padx=4
)


tk.Button(
    controls,
    text="■ STOP",
    command=stop_song,
    bg=RED,
    fg=WHITE,
    font=("Segoe UI", 9, "bold"),
    relief="flat",
    padx=12,
    pady=7,
    cursor="hand2"
).grid(
    row=0,
    column=3,
    padx=4
)


# =========================
# VOLUME
# =========================

tk.Label(
    player_card,
    text="🔊 Volume",
    font=("Segoe UI", 10, "bold"),
    fg=WHITE,
    bg=PANEL
).pack(
    pady=(5, 0)
)


volume_slider = tk.Scale(
    player_card,
    from_=0,
    to=100,
    orient="horizontal",
    length=250,
    command=change_volume,
    bg=PANEL,
    fg=WHITE,
    troughcolor=PANEL_LIGHT,
    highlightthickness=0
)

volume_slider.set(70)

volume_slider.pack()


# =========================
# FOOTER
# =========================

tk.Label(
    main_frame,
    text="♫ Feel the music • Follow the mood ♫",
    font=("Segoe UI", 9),
    fg=MUTED_TEXT,
    bg=BG_DARK
).pack(
    pady=5
)


# =========================
# INITIALIZE PYGAME
# =========================

pygame.mixer.init()

pygame.mixer.music.set_volume(0.7)


# =========================
# START PROGRESS MONITOR
# =========================

root.after(
    500,
    update_progress
)


# =========================
# RUN
# =========================

root.mainloop()