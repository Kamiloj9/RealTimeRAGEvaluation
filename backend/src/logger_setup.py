import logging
from colorama import init, Fore, Style

def setup_logger(
    name: str = 'app_logger',
    level: int = logging.INFO,
    log_file: str = 'app.log'
) -> logging.Logger:
    init(autoreset=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            f'{Fore.GREEN}[%(asctime)s]{Style.RESET_ALL} {Fore.CYAN}%(name)s{Style.RESET_ALL} - '
            f'{Fore.YELLOW}%(levelname)s{Style.RESET_ALL} - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler (no color formatting)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger