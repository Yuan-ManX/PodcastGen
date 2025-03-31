import logging
from dataclasses import dataclass
from typing import List
from youtube_transcript_api import YouTubeTranscriptApi


@dataclass
class TranscriberConfig:
    remove_phrases: List[str] = None


class YouTubeTranscriptFetcher:
    def __init__(self, config: TranscriberConfig = None):
        """
        Initializes the YouTubeTranscriptFetcher with optional configuration.

        Args:
            config (TranscriberConfig, optional): Configuration for the transcriber.
                                                  Defaults to None.
        """
        if config is None:
            config = TranscriberConfig()
        self.config = config
        self.remove_phrases = self.config.remove_phrases or []

    def fetch_transcript(self, url: str) -> str:
        """
        Fetches and cleans the transcript from a YouTube video.

        Args:
            url (str): The URL of the YouTube video.

        Returns:
            str: The cleaned transcript.

        Raises:
            ValueError: If the URL is invalid or the video ID cannot be extracted.
            Exception: For any other exceptions during fetching.
        """
        if not url:
            raise ValueError("URL cannot be empty.")

        try:
            video_id = self._extract_video_id(url)
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            cleaned_transcript = self._clean_transcript(transcript)
            return cleaned_transcript
        except Exception as e:
            logging.error(f"Failed to fetch YouTube transcript: {e}")
            raise

    def _extract_video_id(self, url: str) -> str:
        """
        Extracts the video ID from the YouTube URL.

        Args:
            url (str): The URL of the YouTube video.

        Returns:
            str: The extracted video ID.

        Raises:
            ValueError: If the video ID cannot be extracted.
        """
        if "youtube.com/watch" in url:
            query_params = url.split("?")[1]
            params = query_params.split("&")
            for param in params:
                if param.startswith("v="):
                    return param.split("=")[1]
        elif "youtu.be/" in url:
            return url.split("/")[-1]
        else:
            raise ValueError("Invalid YouTube URL format.")

    def _clean_transcript(self, transcript: List[dict]) -> str:
        """
        Cleans the transcript by removing specified phrases.

        Args:
            transcript (List[dict]): The raw transcript data.

        Returns:
            str: The cleaned transcript.
        """
        filtered_entries = [
            entry['text'] for entry in transcript
            if entry['text'].lower() not in self.remove_phrases
        ]
        return ' '.join(filtered_entries)


def run_transcriber(seed: int = 45) -> None:
    """
    Tests the YouTubeTranscriptFetcher with a specific URL and saves the transcript.

    Args:
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    config = TranscriberConfig(remove_phrases=["[music]", "[laughter]"])
    fetcher = YouTubeTranscriptFetcher(config=config)

    url = ""

    if not url:
        logging.error("URL is empty. Please provide a valid YouTube video URL.")
        return

    try:
        transcript = fetcher.fetch_transcript(url)
        print("Transcript fetched successfully.")

        output_file = 'tests/data/transcripts/youtube_transcript2.txt'
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(transcript)

        print(f"Transcript saved to {output_file}")
        print("First 500 characters of the transcript:")
        print(transcript[:500] + '...' if len(transcript) > 500 else transcript)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return
