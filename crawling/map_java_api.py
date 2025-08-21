import os
import re
import time
from urllib.parse import urljoin, urlparse, parse_qs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, TimeoutException

# 시작 URL
BASE_URL = "https://developers.google.com"
START_URL = "/maps/documentation/javascript/places-js?hl=ko"

# 저장할 폴더 이름
OUTPUT_DIR = "map_javascript_api"

# 결과 저장 폴더 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 방문한 URL 추적 (중복 방지)
visited_urls = set()

# 셀레니움 옵션 설정
chrome_options = Options()
# chrome_options.add_argument("--headless")  # 브라우저 창을 보지 않고 실행하려면 주석 해제
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")

# 로그 메시지 숨기기 옵션들
chrome_options.add_argument("--disable-logging")
chrome_options.add_argument("--disable-logging-redirect")
chrome_options.add_argument("--log-level=3")
chrome_options.add_argument("--silent")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-plugins")
chrome_options.add_argument("--disable-web-security")
chrome_options.add_argument("--disable-features=VizDisplayCompositor")
chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
chrome_options.add_experimental_option('useAutomationExtension', False)

# 추가 로그 제거 옵션
import logging
logging.getLogger('selenium').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# 웹 드라이버 서비스 설정 및 실행
try:
    from webdriver_manager.chrome import ChromeDriverManager
    service = ChromeService(ChromeDriverManager().install())
    print("✅ webdriver-manager로 ChromeDriver 설정 완료")
except ImportError:
    service = ChromeService()
    print("✅ 시스템 ChromeDriver 사용")

driver = webdriver.Chrome(service=service, options=chrome_options)

def normalize_url(url):
    """URL 정규화 (쿼리 파라미터 정렬, 중복 제거용)"""
    if not url:
        return None
    
    # 한국어 파라미터 추가
    if "?hl=ko" not in url and "&hl=ko" not in url:
        if "?" in url:
            url = url + "&hl=ko"
        else:
            url = url + "?hl=ko"
    
    # URL 파싱 후 정규화
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    
    # 쿼리 파라미터 정렬
    normalized_query = "&".join(sorted([f"{k}={v[0]}" for k, v in query_params.items()]))
    
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{normalized_query}"

def expand_sidebar_sections(driver, wait):
    """사이드바의 접힌 섹션들을 모두 펼치기"""
    try:
        print("🔄 사이드바 섹션 확장 중...")
        
        expand_selectors = [
            'button[aria-expanded="false"]',
            '.devsite-nav-toggle',
            'button[aria-controls]',
            '[data-category="referencenav"]',
            '.devsite-nav-item-toggle',
            '.devsite-nav-expandable > button',
            '.devsite-nav-item.devsite-nav-expandable',
            '.devsite-nav-accordion button',
            '.devsite-nav-section button',
            'devsite-book-nav button[aria-expanded="false"]'
        ]
        
        expanded_count = 0
        max_attempts = 10
        
        for attempt in range(max_attempts):
            current_expanded = 0
            for selector in expand_selectors:
                try:
                    buttons = driver.find_elements(By.CSS_SELECTOR, selector)
                    for button in buttons:
                        try:
                            if button.is_displayed() and button.is_enabled():
                                aria_expanded = button.get_attribute('aria-expanded')
                                if aria_expanded == 'false':
                                    driver.execute_script("arguments[0].click();", button)
                                    current_expanded += 1
                                    time.sleep(0.3)
                        except Exception:
                            continue
                except Exception:
                    continue
            
            if current_expanded == 0:
                break  # 더 이상 확장할 섹션이 없음
            
            expanded_count += current_expanded
            time.sleep(1)
        
        print(f"✅ {expanded_count}개 섹션 확장 완료")
        
    except Exception as e:
        print(f"⚠️ 섹션 확장 중 오류: {e}")

def collect_main_tabs(driver):
    """페이지 상단의 주요 탭들 수집 (가이드, 참조, 샘플, 리소스, 기존)"""
    tabs = []
    try:
        print("🎯 상단 주요 탭 수집 중...")
        
        # 상단 탭 선택자들
        tab_selectors = [
            '.devsite-doc-set-nav a',  # 주요 탭 네비게이션
            'nav.devsite-doc-set-nav a',
            '.devsite-tabs a',
            '[role="tablist"] a'
        ]
        
        for selector in tab_selectors:
            try:
                tab_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for tab in tab_elements:
                    href = tab.get_attribute('href')
                    text = (tab.text or '').strip()
                    
                    # Maps JavaScript API 관련 탭만 수집
                    if (href and text and 
                        "developers.google.com/maps/documentation/javascript" in href and
                        text in ['가이드', '참조', '샘플', '리소스', '기존', 'Guides', 'Reference', 'Samples', 'Resources', 'Legacy']):
                        
                        normalized_url = normalize_url(href)
                        if normalized_url:
                            tabs.append({
                                'name': text,
                                'url': normalized_url
                            })
            except Exception as e:
                print(f"⚠️ 탭 선택자 '{selector}' 처리 중 오류: {e}")
                continue
        
        # 중복 제거
        unique_tabs = []
        seen_urls = set()
        for tab in tabs:
            if tab['url'] not in seen_urls:
                seen_urls.add(tab['url'])
                unique_tabs.append(tab)
        
        print(f"✅ {len(unique_tabs)}개 주요 탭 발견:")
        for tab in unique_tabs:
            print(f"   - {tab['name']}: {tab['url']}")
        
        return unique_tabs
        
    except Exception as e:
        print(f"❌ 주요 탭 수집 오류: {e}")
        return []

def collect_sidebar_links(driver, wait, current_tab_name="기본"):
    """현재 페이지의 사이드바에서 모든 링크 수집"""
    try:
        print(f"🔍 [{current_tab_name}] 사이드바 링크 수집 중...")
        
        # 사이드바 확장
        expand_sidebar_sections(driver, wait)
        
        # 사이드바 선택자들
        sidebar_selectors = [
            'devsite-book-nav',
            '.devsite-nav-list',
            '.devsite-section-nav',
            '[role="navigation"]',
            '.devsite-nav',
            'nav.devsite-nav',
            '.devsite-nav-accordion',
            '.devsite-book-nav'
        ]
        
        nav_container = None
        for selector in sidebar_selectors:
            try:
                nav_container = driver.find_element(By.CSS_SELECTOR, selector)
                if nav_container:
                    print(f"✅ [{current_tab_name}] 사이드바 발견: {selector}")
                    break
            except NoSuchElementException:
                continue
        
        if not nav_container:
            print(f"⚠️ [{current_tab_name}] 특정 사이드바를 찾을 수 없음, 전체 페이지에서 검색")
            nav_container = driver.find_element(By.TAG_NAME, "body")
        
        # 모든 링크 수집
        link_elements = nav_container.find_elements(By.TAG_NAME, "a")
        urls_to_crawl = []
        
        for elem in link_elements:
            href = elem.get_attribute("href")
            if (href and 
                "/maps/documentation/javascript" in href and 
                "developers.google.com" in href and
                not href.endswith('#')):  # 앵커 링크 제외
                
                normalized_url = normalize_url(href)
                if normalized_url and normalized_url not in visited_urls:
                    urls_to_crawl.append(normalized_url)
        
        # 중복 제거
        unique_urls = list(dict.fromkeys(urls_to_crawl))
        print(f"✅ [{current_tab_name}] {len(unique_urls)}개의 새로운 링크 수집")
        
        # 디버깅용: 처음 5개 링크 출력
        if unique_urls:
            print(f"   샘플 링크 (처음 5개):")
            for i, url in enumerate(unique_urls[:5], 1):
                print(f"     {i}. {url.split('/')[-1]}")
        
        return unique_urls
        
    except Exception as e:
        print(f"❌ [{current_tab_name}] 사이드바 링크 수집 오류: {e}")
        return []

def process_tabs_in_article(driver, article_element):
    """article 내의 탭 그룹 처리"""
    try:
        tab_groups = article_element.find_elements(By.TAG_NAME, "devsite-selector")
        
        if not tab_groups:
            return article_element.text
        
        print(f"🎯 {len(tab_groups)}개 탭 그룹 발견, 처리 중...")
        
        final_page_text = article_element.text
        
        for tab_group_idx, tab_group in enumerate(tab_groups):
            tab_texts = []
            
            tab_buttons = tab_group.find_elements(
                By.CSS_SELECTOR, "devsite-tabs tab:not(.devsite-overflow-tab)"
            )
            
            if not tab_buttons:
                continue
            
            def get_tab_name(btn):
                txt = (btn.text or "").strip()
                if txt:
                    return txt
                return (
                    btn.get_attribute("aria-controls")
                    or btn.get_attribute("id")
                    or btn.get_attribute("data-tab")
                    or f"UNNAMED_TAB_{tab_group_idx}"
                )
            
            tab_panels = tab_group.find_elements(
                By.CSS_SELECTOR, "section[role='tabpanel']"
            )
            
            panels_by_key = {}
            for p in tab_panels:
                key = p.get_attribute("data-tab")
                if not key:
                    labelledby = p.get_attribute("aria-labelledby") or ""
                    if labelledby.startswith("aria-tab-"):
                        key = labelledby.replace("aria-tab-", "")
                if key:
                    panels_by_key[key] = p
            
            for btn_idx, btn in enumerate(tab_buttons):
                tab_key = (
                    btn.get_attribute("data-tab") 
                    or btn.get_attribute("id") 
                    or f"tab_{btn_idx}"
                )
                tab_name = get_tab_name(btn)
                
                panel_text = ""
                panel = panels_by_key.get(tab_key)
                
                if panel is None:
                    try:
                        driver.execute_script("arguments[0].click();", btn)
                        time.sleep(0.5)
                        panel = tab_group.find_element(
                            By.CSS_SELECTOR,
                            f"section[role='tabpanel'][data-tab='{tab_key}']",
                        )
                    except Exception:
                        panel = None
                
                if panel is not None:
                    try:
                        code_block = panel.find_element(
                            By.CSS_SELECTOR, "pre.devsite-code-highlight"
                        )
                        panel_text = code_block.get_attribute("textContent").strip()
                    except NoSuchElementException:
                        panel_text = (panel.get_attribute("textContent") or "").strip()
                else:
                    panel_text = "(패널을 찾을 수 없음)"
                
                tab_texts.append(f"--- 탭: {tab_name} ---\n{panel_text}")
            
            formatted_tab_content = "\n\n".join(tab_texts)
            if tab_group.text and formatted_tab_content:
                final_page_text = final_page_text.replace(
                    tab_group.text, formatted_tab_content, 1
                )
        
        return final_page_text
        
    except Exception as e:
        print(f"⚠️ 탭 처리 중 오류: {e}")
        return article_element.text

def add_links_to_text(driver, article_element):
    """article 내의 모든 링크에 [URL] 형태로 주소 추가"""
    try:
        links_in_article = article_element.find_elements(By.TAG_NAME, "a")
        link_count = 0
        
        for link in links_in_article:
            href = link.get_attribute("href")
            if href and "javascript:void(0)" not in href and href.startswith("http"):
                try:
                    driver.execute_script(
                        "arguments[0].textContent = arguments[0].textContent.trim() + ' [' + arguments[0].href + ']';",
                        link
                    )
                    link_count += 1
                except Exception:
                    continue
        
        if link_count > 0:
            print(f"🔗 {link_count}개 링크에 URL 주소 추가 완료")
            
    except StaleElementReferenceException:
        print("⚠️ 링크 수정 중 DOM 변경으로 일부 링크 처리 불가")
    except Exception as e:
        print(f"⚠️ 링크 처리 중 오류: {e}")

def crawl_page(url, tab_name="기본"):
    """개별 페이지 크롤링"""
    try:
        print(f"📄 [{tab_name}] 크롤링 중: {url}")
        
        driver.get(url)
        wait = WebDriverWait(driver, 15)
        
        # 페이지 로드 대기
        try:
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(3)  # 페이지 안정화 대기 시간 증가
        except TimeoutException:
            print(f"⚠️ [{tab_name}] 페이지 로드 시간 초과")
            return False
        
        # article 요소 찾기
        try:
            article_element = wait.until(
                EC.presence_of_element_located((By.TAG_NAME, "article"))
            )
        except TimeoutException:
            try:
                article_element = wait.until(
                    EC.presence_of_element_located((By.TAG_NAME, "main"))
                )
            except TimeoutException:
                article_element = driver.find_element(By.TAG_NAME, "body")
        
        # 링크에 URL 주소 추가
        add_links_to_text(driver, article_element)
        
        # 탭 처리 및 최종 텍스트 추출
        final_page_text = process_tabs_in_article(driver, article_element)
        
        # 파일명 생성
        path = url.split("?")[0].replace(BASE_URL, "")
        safe_tab_name = re.sub(r'[/\\?%*:|"<>]', "_", tab_name)
        filename = f"{safe_tab_name}_{re.sub(r'[/\\?%*:|\"<>]', '_', path).strip('_')}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # 내용 구성
        content_to_save = f"Tab: {tab_name}\nSource URL: {url}\n\n{final_page_text}"
        
        # 파일 저장
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content_to_save)
        
        file_size = len(content_to_save)
        print(f"✅ [{tab_name}] 저장 완료: {filename} ({file_size:,} 글자)")
        
        # 방문한 URL 추가
        visited_urls.add(url)
        return True
        
    except Exception as e:
        print(f"❌ [{tab_name}] 페이지 처리 오류: {url} - {e}")
        return False

try:
    # 시작 페이지로 이동
    full_start_url = urljoin(BASE_URL, START_URL)
    print(f"🚀 시작 페이지로 이동: {full_start_url}")
    driver.get(full_start_url)
    
    wait = WebDriverWait(driver, 20)
    
    # 페이지 로드 대기
    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(5)  # 페이지 완전 로드 대기
        print("✅ 페이지 로드 완료")
    except TimeoutException:
        print("⚠️ 페이지 로드 시간 초과, 계속 진행...")
    
    # 전체 URL 수집 리스트
    all_urls_to_crawl = []
    
    # 1. 시작 페이지(기본) 사이드바 링크 수집
    print("\n📋 기본 페이지에서 사이드바 링크 수집...")
    base_urls = collect_sidebar_links(driver, wait, "기본")
    for url in base_urls:
        all_urls_to_crawl.append(("기본", url))
    
    # 2. 상단 주요 탭들 수집
    main_tabs = collect_main_tabs(driver)
    
    # 3. 각 주요 탭별로 이동하여 사이드바 링크 수집
    for tab in main_tabs:
        tab_name = tab['name']
        tab_url = tab['url']
        
        if tab_url in visited_urls:
            print(f"⚠️ [{tab_name}] 이미 방문한 탭, 건너뛰기")
            continue
            
        try:
            print(f"\n🎯 [{tab_name}] 탭으로 이동: {tab_url}")
            driver.get(tab_url)
            time.sleep(3)  # 탭 로드 대기
            
            # 해당 탭에서 사이드바 링크 수집
            tab_urls = collect_sidebar_links(driver, wait, tab_name)
            for url in tab_urls:
                all_urls_to_crawl.append((tab_name, url))
                
            # 탭 자체도 크롤링 대상에 추가
            all_urls_to_crawl.append((tab_name, tab_url))
                
        except Exception as e:
            print(f"❌ [{tab_name}] 탭 처리 오류: {e}")
    
    # 4. 시작 URL도 포함 (맨 앞에)
    all_urls_to_crawl.insert(0, ("기본", normalize_url(full_start_url)))
    
    # 5. 중복 제거 (URL 기준)
    seen_urls = set()
    unique_urls_to_crawl = []
    for tab_name, url in all_urls_to_crawl:
        if url not in seen_urls:
            seen_urls.add(url)
            unique_urls_to_crawl.append((tab_name, url))
    
    print(f"\n" + "=" * 70)
    print(f"🎯 총 {len(unique_urls_to_crawl)}개 페이지 크롤링 시작")
    print("=" * 70)
    
    # 탭별 카운트 출력
    tab_counts = {}
    for tab_name, _ in unique_urls_to_crawl:
        tab_counts[tab_name] = tab_counts.get(tab_name, 0) + 1
    
    print("📊 탭별 페이지 수:")
    for tab_name, count in tab_counts.items():
        print(f"   - {tab_name}: {count}개")
    print("=" * 70)
    
    successful_count = 0
    failed_count = 0
    
    for i, (tab_name, url) in enumerate(unique_urls_to_crawl):
        print(f"\n({i+1}/{len(unique_urls_to_crawl)})")
        
        if crawl_page(url, tab_name):
            successful_count += 1
        else:
            failed_count += 1
        
        # 서버 부하 방지 (대기 시간 약간 증가)
        time.sleep(2)
    
    print("\n" + "=" * 70)
    print("🎉 크롤링 완료!")
    print("=" * 70)
    print(f"✅ 성공: {successful_count}개")
    print(f"❌ 실패: {failed_count}개")
    print(f"📁 저장 폴더: {OUTPUT_DIR}")
    print(f"📊 총 처리: {len(unique_urls_to_crawl)}개")
    
    # 탭별 성공 결과 요약
    print("\n📋 탭별 크롤링 결과:")
    for tab_name, count in tab_counts.items():
        success_count = sum(1 for tn, _ in unique_urls_to_crawl[:successful_count] if tn == tab_name)
        print(f"   - {tab_name}: {success_count}/{count}개 성공")
    
    print("=" * 70)

except Exception as e:
    print(f"💥 치명적 오류 발생: {e}")
    import traceback
    traceback.print_exc()

finally:
    try:
        if 'driver' in locals():
            driver.quit()
    except Exception as e:
        print(f"⚠️ 브라우저 종료 중 오류 (무시 가능): {e}")
    print("\n🔚 브라우저를 종료합니다.")