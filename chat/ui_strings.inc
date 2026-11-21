// The interface's text in every language (included by ui.c). One entry per StrId, the languages in LANG_* order:
// es, en, pt, ru, fr, de, ar, ja, zh-Hans, zh-Hant, ko. pc/export_chat.py reads this file too: the Chinese, Japanese
// and Korean characters found here get glyphs of their own at the hint and title sizes.

static const char* strings[N_STRINGS][N_LANGS] = {
    [S_TAGLINE] = {"%s en la GPU de tu PS4.", "%s on your PS4's GPU.", "%s na GPU do seu PS4.", "%s на GPU вашей PS4.",   // the model's name
                   "%s sur le GPU de votre PS4.", "%s auf der GPU deiner PS4.",
                   "%s على معالج الرسوميات في جهاز PS4 الخاص بك.", "あなたのPS4のGPUで動く%s。",
                   "%s，在你的 PS4 GPU 上运行。", "%s，在您的 PS4 GPU 上執行。", "PS4 GPU에서 실행되는 %s."},
    [S_EMPTY_HINT] = {"100% local y sin internet.", "100% local, no internet.", "100% local e sem internet.",
                      "100% локально, без интернета.", "100 % local, sans Internet.", "100 % lokal, ohne Internet.",
                      "محلي بالكامل وبلا إنترنت.", "完全ローカル、ネット不要。", "100% 本地运行，无需联网。",
                      "100% 本機執行，無需連網。", "100% 로컬, 인터넷 없이."},
    [S_WRITE] = {"Escribir", "Type", "Escrever", "Написать", "Écrire", "Schreiben", "اكتب", "入力", "输入", "輸入", "입력"},
    [S_TALK_HOLD] = {"Mantén para hablar", "Hold to talk", "Segure para falar", "Голос (удерживать)",
                     "Maintenir pour parler", "Zum Sprechen halten", "اضغط مطولًا للتحدث", "長押しで話す", "按住说话",
                     "按住說話", "길게 눌러 말하기"},
    [S_STOP] = {"Detener", "Stop", "Parar", "Остановить", "Arrêter", "Stopp", "إيقاف", "停止", "停止", "停止", "중지"},
    [S_NEW_CHAT] = {"Nueva conversación", "New chat", "Nova conversa", "Новый чат", "Nouvelle discussion", "Neuer Chat",
                    "محادثة جديدة", "新しいチャット", "新对话", "新對話", "새 대화"},
    [S_SCROLL] = {"Desplazar", "Scroll", "Rolar", "Прокрутка", "Défiler", "Scrollen", "تمرير", "スクロール", "滚动", "捲動",
                  "스크롤"},
    [S_TEMPERATURE] = {"Temperatura", "Temperature", "Temperatura", "Температура", "Température", "Temperatur",
                       "درجة الحرارة", "温度", "温度", "溫度", "온도"},
    [S_CONTEXT] = {"contexto", "context", "contexto", "контекст", "contexte", "Kontext", "السياق", "コンテキスト", "上下文",
                   "上下文", "컨텍스트"},
    [S_READING] = {"Leyendo tu mensaje…", "Reading your message…", "Lendo sua mensagem…", "Читаю сообщение…",
                   "Lecture de votre message…", "Lese deine Nachricht…", "أقرأ رسالتك…", "メッセージを読んでいます…",
                   "正在读取你的消息…", "正在讀取您的訊息…", "메시지를 읽는 중…"},
    [S_CONTEXT_FULL] = {"La conversación llenó la memoria del modelo; empiezo una nueva.",
                        "The conversation filled the model's memory; starting a new one.",
                        "A conversa encheu a memória do modelo; começando uma nova.",
                        "Разговор заполнил память модели; начинаю новый.",
                        "La conversation a rempli la mémoire du modèle ; j’en commence une nouvelle.",
                        "Das Gespräch hat den Speicher des Modells gefüllt; ich beginne ein neues.",
                        "امتلأت ذاكرة النموذج بالمحادثة؛ سأبدأ محادثة جديدة.",
                        "会話でモデルのメモリがいっぱいになったので、新しい会話を始めます。",
                        "对话已占满模型的内存，开始新的对话。", "對話已佔滿模型的記憶體，開始新的對話。",
                        "대화가 모델의 메모리를 가득 채워 새 대화를 시작합니다."},
    [S_PLACEHOLDER] = {"Escribe un mensaje…", "Type a message…", "Digite uma mensagem…", "Введите сообщение…",
                       "Écrivez un message…", "Nachricht schreiben…", "اكتب رسالة…", "メッセージを入力…", "输入消息…",
                       "輸入訊息…", "메시지 입력…"},
    [S_KB_TYPE] = {"Escribir", "Type", "Digitar", "Ввод", "Saisir", "Tippen", "كتابة", "入力", "输入", "輸入", "입력"},
    [S_KB_DELETE] = {"Borrar", "Delete", "Apagar", "Удалить", "Effacer", "Löschen", "حذف", "削除", "删除", "刪除", "삭제"},
    [S_KB_SPACE] = {"Espacio", "Space", "Espaço", "Пробел", "Espace", "Leertaste", "مسافة", "スペース", "空格", "空白",
                    "공백"},
    [S_KB_SHIFT] = {"Mayúsculas", "Shift", "Maiúsculas", "Регистр", "Majuscule", "Umschalt", "أحرف كبيرة", "シフト", "大写",
                    "大寫", "시프트"},
    [S_KB_EMOJI] = {"Emoji", "Emoji", "Emoji", "Эмодзи", "Emoji", "Emoji", "رموز تعبيرية", "絵文字", "表情", "表情符號",
                    "이모지"},
    [S_KB_LETTERS] = {"Letras", "Letters", "Letras", "Буквы", "Lettres", "Buchstaben", "الأحرف", "文字", "字母", "字母",
                      "문자"},
    [S_KB_DICTATE] = {"Dictar (mantener)", "Dictate (hold)", "Ditar (segurar)", "Диктовка (удерж.)", "Dicter (maintenir)",
                      "Diktieren (halten)", "إملاء (اضغط مطولًا)", "音声入力（長押し）", "语音输入（按住）", "語音輸入（按住）",
                      "받아쓰기 (길게)"},
    [S_KB_SEND] = {"Enviar", "Send", "Enviar", "Отправить", "Envoyer", "Senden", "إرسال", "送信", "发送", "傳送", "보내기"},
    [S_KB_CLOSE] = {"Cerrar", "Close", "Fechar", "Закрыть", "Fermer", "Schließen", "إغلاق", "閉じる", "关闭", "關閉", "닫기"},
    [S_SPACE_KEY] = {"espacio", "space", "espaço", "пробел", "espace", "Leerzeichen", "مسافة", "スペース", "空格", "空白",
                     "스페이스"},
    [S_SEND_KEY] = {"Enviar", "Send", "Enviar", "Отправить", "Envoyer", "Senden", "إرسال", "送信", "发送", "傳送", "보내기"},
    [S_LISTENING] = {"Escuchando…", "Listening…", "Ouvindo…", "Слушаю…", "J’écoute…", "Ich höre zu…", "أستمع…",
                     "聞いています…", "正在听…", "正在聆聽…", "듣는 중…"},
    [S_RELEASE_TO_END] = {"Suelta para terminar", "Release to finish", "Solte para terminar", "Отпустите, чтобы закончить",
                          "Relâchez pour terminer", "Zum Beenden loslassen", "اترك الزر للإنهاء", "離すと終了",
                          "松开结束", "放開結束", "떼면 끝나요"},
    [S_TRANSCRIBING] = {"Transcribiendo…", "Transcribing…", "Transcrevendo…", "Распознаю речь…", "Transcription…",
                        "Wird transkribiert…", "أحوّل كلامك إلى نص…", "文字に起こしています…", "正在转写…", "正在轉寫…",
                        "받아쓰는 중…"},
    [S_YOUR_MESSAGE] = {"Tu mensaje", "Your message", "Sua mensagem", "Ваше сообщение", "Votre message", "Deine Nachricht",
                        "رسالتك", "あなたのメッセージ", "你的消息", "您的訊息", "내 메시지"},
    [S_SEND] = {"Enviar", "Send", "Enviar", "Отправить", "Envoyer", "Senden", "إرسال", "送信", "发送", "傳送", "보내기"},
    [S_EDIT] = {"Editar", "Edit", "Editar", "Изменить", "Modifier", "Bearbeiten", "تعديل", "編集", "编辑", "編輯", "편집"},
    [S_RETRY] = {"Repetir", "Retry", "Repetir", "Заново", "Réessayer", "Wiederholen", "إعادة", "やり直す", "重试", "重試",
                 "다시"},
    [S_DISCARD] = {"Descartar", "Discard", "Descartar", "Отмена", "Supprimer", "Verwerfen", "تجاهل", "破棄", "放弃", "捨棄",
                   "버리기"},
    [S_NO_MIC] = {"No se detecta micrófono: conecta unos auriculares al mando.",
                  "No microphone found: plug a headset into the controller.",
                  "Nenhum microfone detectado: conecte um fone ao controle.",
                  "Микрофон не найден: подключите гарнитуру к геймпаду.",
                  "Aucun micro détecté : branchez un casque sur la manette.",
                  "Kein Mikrofon gefunden: Schließ ein Headset an den Controller an.",
                  "لم يُعثر على ميكروفون: وصّل سماعة رأس بيد التحكم.",
                  "マイクが見つかりません。コントローラーにヘッドセットをつないでください。",
                  "未找到麦克风：请把耳机插到手柄上。", "找不到麥克風：請將耳機插到控制器上。",
                  "마이크가 없어요. 컨트롤러에 헤드셋을 연결하세요."},
    [S_NO_SPEECH] = {"No se entendió nada. Inténtalo otra vez.", "Didn't catch that. Try again.",
                     "Não entendi nada. Tente de novo.", "Ничего не распознано. Попробуйте ещё раз.",
                     "Je n’ai rien compris. Réessayez.", "Nichts verstanden. Versuch es noch mal.",
                     "لم أفهم شيئًا. حاول مرة أخرى.", "聞き取れませんでした。もう一度どうぞ。", "没听清，请再试一次。",
                     "沒聽清楚，請再試一次。", "알아듣지 못했어요. 다시 해 보세요."},
    [S_ASR_OFF] = {"El reconocimiento de voz no está disponible.", "Speech recognition is not available.",
                   "O reconhecimento de voz não está disponível.", "Распознавание речи недоступно.",
                   "La reconnaissance vocale n’est pas disponible.", "Die Spracherkennung ist nicht verfügbar.",
                   "التعرّف على الكلام غير متاح.", "音声認識は使えません。", "语音识别不可用。", "語音辨識無法使用。",
                   "음성 인식을 사용할 수 없어요."},
    [S_HOLD_LONGER] = {"Mantén □ pulsado mientras hablas.", "Keep □ held while you speak.",
                       "Mantenha □ pressionado enquanto fala.", "Держите □, пока говорите.",
                       "Maintenez □ pendant que vous parlez.", "Halte □ gedrückt, während du sprichst.",
                       "استمر في الضغط على □ أثناء الكلام.", "話している間は□を押し続けてください。",
                       "说话时请一直按住 □。", "說話時請一直按住 □。", "말하는 동안 □를 계속 누르세요."},
    [S_PREPARING] = {"Preparando", "Getting ready", "Preparando", "Подготовка", "Préparation", "Vorbereitung",
                     "جارٍ التحضير", "準備中", "准备中", "準備中", "준비 중"},
    [S_LOADING_MODEL] = {"Cargando el modelo", "Loading the model", "Carregando o modelo", "Загрузка модели",
                         "Chargement du modèle", "Modell wird geladen", "جارٍ تحميل النموذج", "モデルを読み込み中",
                         "正在加载模型", "正在載入模型", "모델 불러오는 중"},
    [S_LOADING_VOICE] = {"Cargando el reconocimiento de voz", "Loading speech recognition",
                         "Carregando o reconhecimento de voz", "Загрузка распознавания речи",
                         "Chargement de la reconnaissance vocale", "Spracherkennung wird geladen",
                         "جارٍ تحميل التعرّف على الكلام", "音声認識を読み込み中", "正在加载语音识别", "正在載入語音辨識",
                         "음성 인식 불러오는 중"},
    [S_ERROR_TITLE] = {"No se pudo iniciar", "Couldn't start", "Não foi possível iniciar", "Не удалось запустить",
                       "Impossible de démarrer", "Start fehlgeschlagen", "تعذّر البدء", "起動できませんでした", "无法启动",
                       "無法啟動", "시작할 수 없어요"},
    [S_ERROR_HINT] = {"Pulsa el botón PS para salir.", "Press the PS button to exit.", "Pressione o botão PS para sair.",
                      "Нажмите кнопку PS, чтобы выйти.", "Appuyez sur la touche PS pour quitter.",
                      "Drück die PS-Taste, um zu beenden.", "اضغط زر PS للخروج.", "PSボタンを押すと終了します。",
                      "按 PS 键退出。", "按 PS 按鈕即可離開。", "PS 버튼을 눌러 종료하세요."},
    [S_OFFLINE] = {"100 % sin conexión", "100% offline", "100% offline", "100% офлайн", "100 % hors ligne",
                   "100 % offline", "بلا إنترنت تمامًا", "完全オフライン", "完全离线", "完全離線", "100% 오프라인"},
    [S_SYSTEM_PROMPT] = {
        "Eres Garlic LLM, un asistente útil que funciona sin internet en la GPU de una PlayStation 4. "
        "Responde en el idioma del usuario, de forma clara y sin rodeos.",
        "You are Garlic LLM, a helpful assistant running offline on a PlayStation 4's GPU. Answer in the "
        "user's language, clearly and directly.",
        "Você é o Garlic LLM, um assistente útil que funciona sem internet na GPU de um PlayStation 4. "
        "Responda no idioma do usuário, de forma clara e direta.",
        "Ты — Garlic LLM, полезный ассистент, который работает без интернета на GPU PlayStation 4. Отвечай на "
        "языке пользователя, ясно и по делу.",
        "Tu es Garlic LLM, un assistant utile qui fonctionne sans Internet sur le GPU d’une PlayStation 4. "
        "Réponds dans la langue de l’utilisateur, de façon claire et directe.",
        "Du bist Garlic LLM, ein hilfreicher Assistent, der ohne Internet auf der GPU einer PlayStation 4 läuft. "
        "Antworte in der Sprache des Nutzers, klar und direkt.",
        "أنت Garlic LLM، مساعد مفيد يعمل دون إنترنت على معالج الرسوميات في جهاز PlayStation 4. "
        "أجب بلغة المستخدم، بوضوح ومباشرة.",
        "あなたは Garlic LLM です。PlayStation 4 の GPU 上でインターネットなしで動く、頼りになるアシスタントです。"
        "ユーザーと同じ言語で、わかりやすく率直に答えてください。",
        "你是 Garlic LLM，一个在 PlayStation 4 的 GPU 上离线运行的实用助手。请用用户使用的语言回答，清楚直接。",
        // (the model drifts into Simplified characters unless asked for Traditional ones)
        "你是 Garlic LLM，一個在 PlayStation 4 的 GPU 上離線執行的實用助理。請用使用者的語言回答，清楚直接；"
        "用中文回答時，請使用繁體中文。",
        "당신은 Garlic LLM입니다. 인터넷 없이 PlayStation 4의 GPU에서 실행되는 유용한 어시스턴트입니다. "
        "사용자의 언어로 명확하고 간결하게 답하세요."},
    [S_LOCAL_GPU] = {"GPU local", "local GPU", "GPU local", "локальный GPU", "GPU local", "lokale GPU", "GPU محلي",
                     "ローカルGPU", "本地 GPU", "本機 GPU", "로컬 GPU"},
    [S_PY_RUN] = {"Ejecutar Python", "Run Python", "Executar Python", "Запустить Python", "Lancer Python",
                  "Python starten", "تشغيل Python", "Pythonを実行", "运行 Python", "執行 Python", "Python 실행"},
    [S_PY_EXIT] = {"Salir", "Exit", "Sair", "Выйти", "Quitter", "Beenden", "خروج", "終了", "退出", "離開", "나가기"},
    [S_PY_CLICK] = {"Clic", "Click", "Clique", "Клик", "Clic", "Klick", "نقر", "クリック", "点击", "點擊", "클릭"},
    [S_PY_RCLICK] = {"Clic derecho", "Right click", "Clique direito", "Правый клик", "Clic droit", "Rechtsklick",
                     "نقر أيمن", "右クリック", "右键", "右鍵", "오른쪽 클릭"},
    [S_PY_MOVE] = {"Mover", "Move", "Mover", "Двигать", "Déplacer", "Bewegen", "تحريك", "移動", "移动", "移動", "이동"},
    [S_PY_KEYS] = {"Flechas", "Arrows", "Setas", "Стрелки", "Flèches", "Pfeiltasten", "الأسهم", "矢印キー", "方向键",
                   "方向鍵", "방향키"},
    [S_PY_BACK] = {"Volver al chat", "Back to chat", "Voltar ao chat", "Назад в чат", "Retour au chat", "Zurück zum Chat",
                   "العودة إلى المحادثة", "チャットに戻る", "返回对话", "返回對話", "대화로 돌아가기"},
    [S_PY_AGAIN] = {"Ejecutar de nuevo", "Run again", "Executar de novo", "Запустить снова", "Relancer", "Neu starten",
                    "تشغيل مرة أخرى", "もう一度実行", "再次运行", "再次執行", "다시 실행"},
    [S_PY_FIX] = {"Pedir arreglo", "Ask for a fix", "Pedir correção", "Попросить исправить", "Demander une correction",
                  "Um Korrektur bitten", "طلب إصلاح", "修正を頼む", "请求修复", "請求修正", "수정 요청"},
    [S_PY_DONE] = {"Programa terminado", "Program finished", "Programa terminado", "Программа завершена",
                   "Programme terminé", "Programm beendet", "انتهى البرنامج", "プログラムが終了しました", "程序已结束",
                   "程式已結束", "프로그램이 끝났어요"},
    [S_PY_ERROR] = {"El programa tuvo un error", "The program hit an error", "O programa teve um erro", "В программе ошибка",
                    "Le programme a rencontré une erreur", "Das Programm hatte einen Fehler", "حدث خطأ في البرنامج",
                    "プログラムでエラーが起きました", "程序出错了", "程式發生錯誤", "프로그램에 오류가 났어요"},
    [S_PY_STOPPED] = {"Programa detenido", "Program stopped", "Programa interrompido", "Программа остановлена",
                      "Programme arrêté", "Programm gestoppt", "أُوقف البرنامج", "プログラムを止めました", "程序已停止",
                      "程式已停止", "프로그램을 멈췄어요"},
    [S_PY_STARTING] = {"Ejecutando", "Running", "Executando", "Запуск", "Lancement de", "Starte", "تشغيل", "実行中",
                       "正在运行", "正在執行", "실행 중"},
    // "Ask for a fix": states that the program ran on the PS4 itself; without that context the model may answer
    // that Python cannot run on a PS4
    [S_PY_FIX_MSG] = {
        "Ejecuté tu programa aquí mismo, en la PS4, y falló con este error:",
        "I ran your program right here on the PS4, and it failed with this error:",
        "Executei o seu programa aqui mesmo, no PS4, e ele falhou com este erro:",
        "Я запустил твою программу прямо здесь, на PS4, и она завершилась с ошибкой:",
        "J’ai exécuté ton programme ici même, sur la PS4, et il a échoué avec cette erreur :",
        "Ich habe dein Programm hier auf der PS4 ausgeführt, und es ist mit diesem Fehler abgebrochen:",
        "شغّلت برنامجك هنا على جهاز PS4 نفسه، وفشل بهذا الخطأ:",
        "このPS4上であなたのプログラムを実行したところ、次のエラーで止まりました：",
        "我就在这台 PS4 上运行了你的程序，它出现了这个错误：",
        "我就在這台 PS4 上執行了你的程式，它出現了這個錯誤：",
        "이 PS4에서 직접 당신의 프로그램을 실행했는데, 다음 오류로 실패했습니다:"},
    // "Ask for a fix", the request: lists what Garlic's Python has and what it lacks
    [S_PY_FIX_ASK] = {
        "Corrígelo para el Python 3 de Garlic (MicroPython): hay turtle, pygame y tkinter adaptados a la tele "
        "(las flechas, Espacio y Enter van al mando), input(), print(), random, math, time, collections, "
        "itertools, string y copy; no hay numpy, matplotlib, archivos, red ni sonido. Casi siempre el fallo "
        "está en el propio código: busca la causa y corrígela sin quitar nada de lo pedido. Escribe el "
        "programa completo en un solo bloque ```python.",
        "Fix it for Garlic's Python 3 (MicroPython): it has turtle, pygame and tkinter adapted to the TV "
        "(arrows, Space and Enter are on the controller), input(), print(), random, math, time, collections, "
        "itertools, string and copy; there is no numpy, matplotlib, files, network or sound. The bug is "
        "almost always in the code itself: find the cause and fix it without dropping anything that was asked "
        "for. Write the whole program in a single ```python block.",
        "Corrija-o para o Python 3 do Garlic (MicroPython): há turtle, pygame e tkinter adaptados à TV (as "
        "setas, Espaço e Enter ficam no controle), input(), print(), random, math, time, collections, "
        "itertools, string e copy; não há numpy, matplotlib, arquivos, rede nem som. Quase sempre a falha "
        "está no próprio código: encontre a causa e corrija sem tirar nada do que foi pedido. Escreva o "
        "programa completo em um único bloco ```python.",
        "Исправь её для Python 3 в Garlic (MicroPython): есть turtle, pygame и tkinter для телевизора "
        "(стрелки, пробел и Enter — на геймпаде), input(), print(), random, math, time, collections, "
        "itertools, string и copy; нет numpy, matplotlib, файлов, сети и звука. Почти всегда ошибка в самом "
        "коде: найди причину и исправь её, ничего не убирая из того, что просили. Напиши всю программу "
        "целиком в одном блоке ```python.",
        "Corrige-le pour le Python 3 de Garlic (MicroPython) : il y a turtle, pygame et tkinter adaptés à la "
        "télé (les flèches, Espace et Entrée sont sur la manette), input(), print(), random, math, time, "
        "collections, itertools, string et copy ; il n’y a pas de numpy, de matplotlib, de fichiers, de réseau "
        "ni de son. L’erreur est presque toujours dans le code lui-même : trouve la cause et corrige-la sans "
        "rien retirer de ce qui était demandé. Écris le programme complet dans un seul bloc ```python.",
        "Korrigiere es für das Python 3 von Garlic (MicroPython): Es gibt turtle, pygame und tkinter, angepasst an "
        "den Fernseher (Pfeile, Leertaste und Enter liegen auf dem Controller), input(), print(), random, math, "
        "time, collections, itertools, string und copy; es gibt kein numpy, kein matplotlib, keine Dateien, kein "
        "Netzwerk und keinen Ton. Der Fehler liegt fast immer im Code selbst: Finde die Ursache und behebe sie, "
        "ohne etwas wegzulassen, das verlangt wurde. Schreib das ganze Programm in einen einzigen ```python-Block.",
        "أصلحه ليعمل على Python 3 في Garlic (MicroPython): تتوفر turtle وpygame وtkinter مهيأة للتلفاز (الأسهم "
        "والمسافة وEnter على يد التحكم)، وinput() وprint() وrandom وmath وtime وcollections وitertools وstring "
        "وcopy؛ ولا تتوفر numpy ولا matplotlib ولا الملفات ولا الشبكة ولا الصوت. يكون الخطأ غالبًا في الكود "
        "نفسه: ابحث عن السبب وأصلحه دون حذف أي شيء مما طُلب. اكتب البرنامج كاملًا في كتلة ```python واحدة.",
        "Garlic の Python 3（MicroPython）向けに直してください。テレビ向けに調整した turtle、pygame、tkinter"
        "（矢印キー、スペース、Enter はコントローラーにあります）、input()、print()、random、math、time、"
        "collections、itertools、string、copy が使えます。numpy、matplotlib、ファイル、ネットワーク、音はありません。"
        "エラーの原因はほとんどの場合コード自体にあります。原因を見つけて、頼まれた内容を削らずに直してください。"
        "プログラム全体を1つの ```python ブロックで書いてください。",
        "请把它改成适用于 Garlic 的 Python 3（MicroPython）：有为电视调整过的 turtle、pygame 和 tkinter（方向键、"
        "空格和回车在手柄上），以及 input()、print()、random、math、time、collections、itertools、string 和 copy；"
        "没有 numpy、matplotlib、文件、网络和声音。错误几乎总是出在代码本身：找出原因并修复，不要删掉任何要求的"
        "内容。把完整的程序写在一个 ```python 代码块里。",
        "請把它改成適用於 Garlic 的 Python 3（MicroPython）：有為電視調整過的 turtle、pygame 和 tkinter（方向鍵、"
        "空白鍵和 Enter 在控制器上），以及 input()、print()、random、math、time、collections、itertools、string 和 "
        "copy；沒有 numpy、matplotlib、檔案、網路和聲音。錯誤幾乎總是出在程式碼本身：找出原因並修正，不要刪掉任何"
        "要求的內容。把完整的程式寫在一個 ```python 區塊裡，並使用繁體中文。",
        "Garlic의 Python 3(MicroPython)에 맞게 고쳐 주세요. TV에 맞춘 turtle, pygame, tkinter(방향키, 스페이스, "
        "Enter는 컨트롤러에 있음), input(), print(), random, math, time, collections, itertools, string, copy가 "
        "있고, numpy, matplotlib, 파일, 네트워크, 소리는 없습니다. 오류는 거의 항상 코드 자체에 있습니다. 원인을 "
        "찾아서 요청한 내용을 빼지 말고 고치세요. 전체 프로그램을 하나의 ```python 블록으로 작성하세요."},
    [S_PY_INPUT] = {"Responde al programa…", "Answer the program…", "Responda ao programa…", "Ответьте программе…",
                    "Répondez au programme…", "Antworte dem Programm…", "أجب البرنامج…", "プログラムに答える…",
                    "回答程序…", "回答程式…", "프로그램에 답하기…"},
    [S_VOICE_MODE] = {"Modo voz", "Voice mode", "Modo voz", "Голосовой режим", "Mode vocal", "Sprachmodus",
                      "الوضع الصوتي", "音声モード", "语音模式", "語音模式", "음성 모드"},
    [S_THINKING] = {"Pensando…", "Thinking…", "Pensando…", "Думаю…", "Réflexion…", "Denke nach…", "أفكر…", "考え中…",
                    "思考中…", "思考中…", "생각 중…"},
    [S_LISTENING_SHORT] = {"Te escucho…", "Listening…", "Estou ouvindo…", "Слушаю…", "Je vous écoute…", "Ich höre…",
                           "أسمعك…", "聞いています…", "我在听…", "我在聽…", "듣고 있어요…"},
    [S_VOICE_LOADING] = {"Preparando la voz…", "Getting the voice ready…", "Preparando a voz…", "Готовлю голос…",
                         "Préparation de la voix…", "Stimme wird vorbereitet…", "جارٍ تجهيز الصوت…", "音声を準備中…",
                         "正在准备语音…", "正在準備語音…", "음성 준비 중…"},
    [S_VOICE_NO_TTS] = {"La voz no está disponible: Garlic responderá con texto",
                        "The voice is not available: Garlic will answer in text",
                        "A voz não está disponível: o Garlic responderá em texto",
                        "Голос недоступен: Garlic ответит текстом",
                        "La voix n’est pas disponible : Garlic répondra par écrit",
                        "Die Stimme ist nicht verfügbar: Garlic antwortet mit Text",
                        "الصوت غير متاح: سيرد Garlic كتابةً", "音声は使えません。Garlic はテキストで答えます",
                        "语音不可用：Garlic 将用文字回答", "語音無法使用：Garlic 將用文字回答",
                        "음성을 쓸 수 없어요. Garlic이 글로 답해요"},
    // "Think" (R3)
    [S_THINK_OFF_HINT] = {"Pensar: no", "Think: off", "Pensar: não", "Думать: нет", "Réfléchir : non", "Denken: aus",
                          "التفكير: لا", "思考：オフ", "思考：关", "思考：關", "생각: 끔"},
    [S_THINK_ON_HINT] = {"Pensar: sí", "Think: on", "Pensar: sim", "Думать: да", "Réfléchir : oui", "Denken: an",
                         "التفكير: نعم", "思考：オン", "思考：开", "思考：開", "생각: 켬"},
    [S_THINK_ON_TOAST] = {"Pensar activado: más acierto en preguntas difíciles, pero tarda ~1 min",
                          "Thinking on: better at hard questions, but takes ~1 min",
                          "Pensar ativado: acerta mais em perguntas difíceis, mas demora ~1 min",
                          "Размышление включено: точнее в трудных вопросах, но около минуты",
                          "Réflexion activée : plus juste sur les questions difficiles, mais ~1 min",
                          "Denken an: besser bei schweren Fragen, dauert aber ~1 Min.",
                          "التفكير مفعّل: أدق في الأسئلة الصعبة، لكنه يستغرق نحو دقيقة",
                          "思考オン：難しい質問に強くなりますが、1分ほどかかります",
                          "思考已开启：难题答得更准，但要约 1 分钟", "思考已開啟：難題答得更準，但需約 1 分鐘",
                          "생각 켬: 어려운 질문에 더 정확하지만 1분쯤 걸려요"},
    [S_THINK_OFF_TOAST] = {"Pensar desactivado: respuestas al momento", "Thinking off: instant answers",
                           "Pensar desativado: respostas na hora", "Размышление выключено: ответы сразу",
                           "Réflexion désactivée : réponses immédiates", "Denken aus: sofortige Antworten",
                           "التفكير معطّل: إجابات فورية", "思考オフ：すぐに答えます", "思考已关闭：立即回答",
                           "思考已關閉：立即回答", "생각 끔: 바로 답해요"},
    [S_THOUGHT] = {"Pensó %d s · %d tokens", "Thought for %d s · %d tokens", "Pensou %d s · %d tokens",
                   "Думал %d с · токенов: %d", "A réfléchi %d s · %d jetons", "%d s nachgedacht · %d Tokens",
                   "فكّر %d ث · %d توكن", "%d秒思考 · %dトークン", "思考了 %d 秒 · %d 个 token",
                   "思考了 %d 秒 · %d 個 token", "%d초 생각함 · 토큰 %d개"},
    [S_THINK_CUT] = {"tope alcanzado", "limit reached", "limite atingido", "лимит", "limite atteinte", "Limit erreicht",
                     "بلغ الحد", "上限に到達", "已达上限", "已達上限", "한도 도달"},
    [S_THINK_CHIP] = {"Pensar", "Think", "Pensar", "Думать", "Réflexion", "Denken", "تفكير", "思考", "思考", "思考", "생각"},
    // The OpenAI-compatible API (L3). The toast that turns it on is followed by its address: "API on: http://..."
    [S_API_ON] = {"API encendida", "API on", "API ligada", "API включён", "API activée", "API an", "API مفعّلة",
                  "API オン", "API 已开启", "API 已開啟", "API 켬"},
    [S_API_OFF] = {"API apagada", "API off", "API desligada", "API выключен", "API désactivée", "API aus", "API معطّلة",
                   "API オフ", "API 已关闭", "API 已關閉", "API 끔"},
    [S_API_FAILED] = {"La API no pudo iniciarse: revisa la red de la consola",
                      "The API could not start: check the console's network",
                      "A API não pôde iniciar: verifique a rede do console",
                      "Не удалось запустить API: проверьте сеть консоли",
                      "L’API n’a pas pu démarrer : vérifiez le réseau de la console",
                      "Die API konnte nicht starten: Prüf das Netzwerk der Konsole",
                      "تعذّر تشغيل API: تحقّق من شبكة الجهاز",
                      "APIを起動できませんでした。本体のネットワークを確認してください",
                      "API 无法启动：请检查主机的网络", "API 無法啟動：請檢查主機的網路",
                      "API를 시작할 수 없어요. 본체 네트워크를 확인하세요"},
    [S_API_BUSY] = {"Respondiendo a una petición de la API…", "Answering an API request…",
                    "Respondendo a uma requisição da API…", "Отвечаю на запрос API…", "Réponse à une requête API…",
                    "Beantworte eine API-Anfrage…", "أرد على طلب API…", "APIリクエストに答えています…",
                    "正在回答 API 请求…", "正在回應 API 請求…", "API 요청에 답하는 중…"},
    [S_API_DONE] = {"Respondí una petición de la API; empiezo una conversación nueva.",
                    "I answered an API request; starting a new conversation.",
                    "Respondi a uma requisição da API; começando uma nova conversa.",
                    "Ответил на запрос API; начинаю новый разговор.",
                    "J’ai répondu à une requête API ; je commence une nouvelle conversation.",
                    "Ich habe eine API-Anfrage beantwortet; ich beginne ein neues Gespräch.",
                    "أجبت عن طلب API؛ سأبدأ محادثة جديدة.", "APIリクエストに答えました。新しい会話を始めます。",
                    "已回答一个 API 请求，开始新的对话。", "已回應一個 API 請求，開始新的對話。",
                    "API 요청에 답했어요. 새 대화를 시작합니다."},
    // OPTIONS on the keyboard: the console's own keyboard, with its input methods (Chinese, Japanese, Korean)
    [S_KB_SYSTEM] = {"Teclado del sistema", "System keyboard", "Teclado do sistema", "Системная клавиатура",
                     "Clavier du système", "Systemtastatur", "لوحة مفاتيح النظام", "システムキーボード", "系统键盘",
                     "系統鍵盤", "시스템 키보드"},
};

// The titles of the keyboard's emoji pages
static const char* emoji_titles[N_LANGS][7] = {
    {"Caras", "Gestos y personas", "Animales y naturaleza", "Comida y bebida", "Actividades y viajes", "Objetos",
     "Corazones y símbolos"},
    {"Smileys", "Gestures & people", "Animals & nature", "Food & drink", "Activities & travel", "Objects",
     "Hearts & symbols"},
    {"Carinhas", "Gestos e pessoas", "Animais e natureza", "Comida e bebida", "Atividades e viagens", "Objetos",
     "Corações e símbolos"},
    {"Смайлики", "Жесты и люди", "Животные и природа", "Еда и напитки", "Занятия и путешествия", "Предметы",
     "Сердца и символы"},
    {"Smileys", "Gestes et personnes", "Animaux et nature", "Nourriture et boissons", "Activités et voyages", "Objets",
     "Cœurs et symboles"},
    {"Smileys", "Gesten und Personen", "Tiere und Natur", "Essen und Trinken", "Aktivitäten und Reisen", "Objekte",
     "Herzen und Symbole"},
    {"الوجوه", "الإيماءات والأشخاص", "الحيوانات والطبيعة", "الطعام والشراب", "الأنشطة والسفر", "الأشياء",
     "القلوب والرموز"},
    {"顔", "ジェスチャーと人", "動物と自然", "食べ物と飲み物", "アクティビティと旅行", "もの", "ハートと記号"},
    {"笑脸", "手势和人物", "动物和自然", "食物和饮料", "活动和旅行", "物品", "爱心和符号"},
    {"笑臉", "手勢和人物", "動物和自然", "食物和飲料", "活動和旅行", "物品", "愛心和符號"},
    {"얼굴", "손짓과 사람", "동물과 자연", "음식과 음료", "활동과 여행", "사물", "하트와 기호"},
};

const char* const lang_codes[N_LANGS] = {"es", "en", "pt", "ru", "fr", "de", "ar", "ja", "zh-Hans", "zh-Hant", "ko"};
