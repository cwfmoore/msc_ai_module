import os
from datetime import datetime as dt
from datetime import timedelta as td
from datetime import timezone as tz
from rich import print
import tomllib
import os
from typing import Dict, Any
import pandas as pd


class Tools:

    def __init__(self):
        pass

    def print_message(self, message_type, message='', format_dict=None, **kwargs):
        # Check the 'logging' argument and return immediately if logging is disabled
        if not kwargs.get('logging', True):
            return

        message_types = {
            'error': '❌ [red bold]ERROR!',
            'success': '✅ [green bold]SUCCESS!',
            'time_taken': '⏱️ [blue bold]TIME TAKEN',
            'attention': '🔔 [hot_pink bold]ATTENTION!',
            'working': '⚙️ [blue bold]WORKING',
            'testing': '🧪 [hot_pink bold]TESTING',
            'warning': '⚠️ [red bold]! WARNING !',
            'waiting': '⏳ [magenta bold]Waiting...',
            'info': 'ℹ️ [blue bold]INFO...'
        }
        message_type = message_types.get(message_type.lower(), message_type)

        if kwargs.get('new_line_before'):
            print('')

        formatted_message = message
        if format_dict:
            formatted_message = message + ' '
            for key, value in format_dict.items():
                formatted_key = f"[magenta]{key.upper().replace('_', ' ')}[yellow]:"
                formatted_value = f"[cyan]{value}[yellow], "
                formatted_message += f"{formatted_key} {formatted_value}"
            formatted_message = formatted_message[:-2]

        print(f'>>> {message_type}[white]: {formatted_message}')

        if kwargs.get('new_line_after'):
            print('')

    def get_now(self):
        now = dt.now(tz=tz.utc).strftime('%H:%M:%S')
        return now
    
    def load_toml_file(self, file_path: str) -> Dict[Any, Any]:
        """
        Load a TOML file. If the file doesn't exist, create an empty one.
        
        Args:
            file_path: Path to the TOML file
            
        Returns:
            Dictionary containing the TOML file data
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                
                # Create empty TOML file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("# Empty TOML file\n")
                return {}
            
            # File exists, load it
            with open(file_path, 'rb') as f:
                return tomllib.load(f)
                
        except Exception as e:
            print(f"Error loading TOML file: {e}")
            return {}

    def correct_header_spelling(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_dict = {'nacionality': 'nationality'}
        if 'nacionality' in df.columns:
            df.rename(columns=rename_dict, inplace=True)
        return df                       

    def normalise_columns(self, df):
        def snake_case(text):
            text = '_'.join(' '.join(str(text).split()).lower().split())
            text = text.replace('/', '_')
            text = text.replace('(', '')
            text = text.replace(')', '')
            text = text.replace('\'', '')
            return '_'.join(' '.join(str(text).split()).lower().split())
        df = df.copy()
        df.columns = [snake_case(col) for col in df.columns]
        return df

    def load_dataset(self, file_name: str, process_headers: bool = True, correct_header_spelling: bool = True) -> pd.DataFrame:
        try:
            file_path = os.path.join(os.getcwd(), 'datasets', file_name)
            df = pd.read_csv(file_path, delimiter=';', encoding='utf-8', low_memory=False)
            if process_headers:
                df = self.normalise_columns(df)
            if correct_header_spelling:
                df = self.correct_header_spelling(df)
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return pd.DataFrame()

if __name__ == '__main__':
    tools = Tools()
    df = tools.load_dataset('dataset_raw.csv')
    print(df.head())
    


