import os
import queue
import threading
import time

from selenium import webdriver

from integration.data.config import CHROME_PATH, EXTENSIONS_PATH


class ChromeDriverPool:
    def __init__(self, max_drivers=4):
        self.max_drivers = max_drivers
        self.available_drivers = queue.Queue()
        self.used_drivers = set()
        self._lock = threading.RLock()

    def get_driver(self):
        with self._lock:
            if not self.available_drivers.empty():
                driver = self.available_drivers.get()
                self.used_drivers.add(driver)
                return driver

            if len(self.used_drivers) < self.max_drivers:
                driver = self._create_new_driver()
                self.used_drivers.add(driver)
                return driver

            # Wait for a driver to be returned
            while self.available_drivers.empty():
                time.sleep(0.1)
            driver = self.available_drivers.get()
            self.used_drivers.add(driver)
            return driver

    def return_driver(self, driver):
        with self._lock:
            if driver in self.used_drivers:
                self.used_drivers.remove(driver)

                try:
                    # Just close all new tabs that were opened during scraping
                    # Keep the original tab intact without trying to clear storage
                    handles = driver.window_handles

                    if len(handles) > 1:
                        # Remember the first tab (usually the data: URL tab)
                        original_handle = handles[0]

                        # Close all other tabs
                        for handle in handles[1:]:
                            try:
                                driver.switch_to.window(handle)
                                driver.close()
                            except Exception as e:
                                print(f"Error closing tab: {e}")

                        # Go back to the original tab
                        driver.switch_to.window(original_handle)

                    # Only clear cookies, don't touch localStorage or sessionStorage
                    try:
                        driver.delete_all_cookies()
                    except Exception as e:
                        print(f"Error clearing cookies: {e}")

                    # Add back to available pool
                    self.available_drivers.put(driver)
                except Exception as e:
                    print(f"Error cleaning up driver, will create a new one: {e}")
                    try:
                        driver.quit()  # Try to properly close the driver
                    except:
                        pass  # Ignore errors during quit

                    # Create and add a fresh driver
                    try:
                        new_driver = self._create_new_driver()
                        self.available_drivers.put(new_driver)
                    except Exception as new_e:
                        print(f"Failed to create replacement driver: {new_e}")

    def _create_new_driver(self):
        # Create a new Chrome driver with all the needed options
        options = webdriver.ChromeOptions()
        options.binary_location = CHROME_PATH

        # Add all the options from init_web_driver()
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-infobars')
        options.add_argument('--disable-gpu')
        options.add_argument(
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36')

        # DANGER ZONE settings
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--allow-insecure-localhost')
        options.add_argument('--ignore-urlfetcher-cert-requests')
        options.add_argument('--disable-net-security')

        # Configure experimental options
        options.add_experimental_option('useAutomationExtension', False)
        options.add_experimental_option('excludeSwitches', ['enable-automation'])

        # Load extensions
        options.add_extension(os.path.join(EXTENSIONS_PATH, 'chrome', 'nopecha.crx'))
        options.add_extension(os.path.join(EXTENSIONS_PATH, 'chrome', 'accept-all-cookies.crx'))

        return webdriver.Chrome(options=options)

    def close_all(self):
        with self._lock:
            while not self.available_drivers.empty():
                driver = self.available_drivers.get()
                try:
                    driver.quit()
                except:
                    pass

            for driver in list(self.used_drivers):
                try:
                    driver.quit()
                except:
                    pass
            self.used_drivers.clear()
