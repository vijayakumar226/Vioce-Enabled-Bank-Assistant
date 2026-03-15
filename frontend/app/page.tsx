"use client";

import Image from "next/image";
import { useCallback, useEffect, useRef, useState, type ChangeEvent } from "react";

type Attachment = {
  id: string;
  name: string;
  mimeType: string;
  preview?: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  extractedText?: string;
  imageUrl?: string;
  imageName?: string;
  messageType?: "text" | "image" | "pending_image";
  pendingFile?: File | null;
  attachments?: Attachment[];
};

type Chat = {
  id: string;
  title: string;
  sessionId: string;
  messages: Message[];
  pinned?: boolean;
  archived?: boolean;
};

type UploadedAttachmentResponse = {
  id: string;
  name: string;
  mime_type: string;
  preview?: string;
};

type ImageChatResponse = {
  answer: string;
  extracted_text?: string;
};

type SpeechRecognitionInstance = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start: () => void;
  stop: () => void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
};

type SpeechRecognitionConstructor = new () => SpeechRecognitionInstance;

type SpeechRecognitionEventLike = {
  resultIndex: number;
  results: ArrayLike<{
    isFinal: boolean;
    0: {
      transcript: string;
    };
  }>;
};

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  }
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";
const CHAT_STORAGE_KEY = "bank_rag_chats";
const ACTIVE_CHAT_STORAGE_KEY = "bank_rag_active_chat";

function createId() {
  return crypto.randomUUID();
}

function createAssistantMessage(content: string): Message {
  return {
    id: createId(),
    role: "assistant",
    content,
    messageType: "text",
  };
}

function createChat(): Chat {
  return {
    id: createId(),
    title: "New chat",
    sessionId: `sess_${createId()}`,
    messages: [
      createAssistantMessage(
        "Hello! I am your SBI Banking Assistant.\n\nI can help you with loans, accounts, charges, complaints and more.\n\nWhat would you like to know today?"
      ),
    ],
  };
}

function makeTitle(text: string) {
  const trimmed = text.trim();
  if (!trimmed) {
    return "New chat";
  }
  return trimmed.length > 48 ? `${trimmed.slice(0, 48).trimEnd()}...` : trimmed;
}

function normalizeChats(raw: Chat[]): Chat[] {
  return raw.map((chat) => {
    const firstUserMessage = chat.messages.find((message) => message.role === "user");
    const normalizedTitle =
      firstUserMessage && (!chat.title || /^new chat$/i.test(chat.title))
        ? makeTitle(firstUserMessage.content)
        : chat.title || "New chat";

    return {
      ...chat,
      title: normalizedTitle,
      pinned: chat.pinned ?? false,
      archived: chat.archived ?? false,
      messages: chat.messages.map((message) => ({
        ...message,
        id: message.id || createId(),
        messageType:
          message.messageType ?? (message.imageUrl ? "image" : "text"),
        pendingFile: null,
        attachments: message.attachments ?? [],
      })),
    };
  });
}

function serializeChats(chats: Chat[]): Chat[] {
  return chats.map((chat) => ({
    ...chat,
    messages: chat.messages.map((message) => ({
      ...message,
      pendingFile: null,
    })),
  }));
}

export default function Page() {
  const [mounted, setMounted] = useState(false);
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [voiceOn, setVoiceOn] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [pendingAttachments, setPendingAttachments] = useState<Attachment[]>([]);
  const [attachmentError, setAttachmentError] = useState<string | null>(null);
  const [speechSupported, setSpeechSupported] = useState(false);
  const [openMenuChatId, setOpenMenuChatId] = useState<string | null>(null);

  const bottomRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const streamRef = useRef<AbortController | null>(null);
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef("");
  const chatsRef = useRef<Chat[]>([]);
  const activeChatIdRef = useRef<string | null>(null);
  const voiceOnRef = useRef(false);

  useEffect(() => {
    chatsRef.current = chats;
  }, [chats]);

  useEffect(() => {
    activeChatIdRef.current = activeChatId;
  }, [activeChatId]);

  useEffect(() => {
    inputRef.current = input;
  }, [input]);

  useEffect(() => {
    voiceOnRef.current = voiceOn;
  }, [voiceOn]);

  const speak = useCallback((text: string) => {
    if (!voiceOnRef.current) return;
    if (typeof window === "undefined") return;
    if (!("speechSynthesis" in window)) return;

    const clean = text.trim();
    if (!clean) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(clean);
    window.speechSynthesis.speak(utterance);
  }, []);

  const updateChat = useCallback((chatId: string, updater: (chat: Chat) => Chat) => {
    setChats((prev) => prev.map((chat) => (chat.id === chatId ? updater(chat) : chat)));
  }, []);

  const ensureChat = useCallback(() => {
    const currentChatId = activeChatIdRef.current;
    const currentChat = chatsRef.current.find((chat) => chat.id === currentChatId);
    if (currentChat) {
      return currentChat;
    }

    const nextChat = createChat();
    setChats((prev) => [nextChat, ...prev]);
    setActiveChatId(nextChat.id);
    chatsRef.current = [nextChat, ...chatsRef.current];
    activeChatIdRef.current = nextChat.id;
    return nextChat;
  }, []);

  useEffect(() => {
    setMounted(true);

    const storedChats = localStorage.getItem(CHAT_STORAGE_KEY);
    const parsedChats = storedChats ? normalizeChats(JSON.parse(storedChats) as Chat[]) : [];
    const freshChat = createChat();
    const nextChats = [freshChat, ...parsedChats];
    const nextActiveChatId = freshChat.id;

    setChats(nextChats);
    setActiveChatId(nextActiveChatId);
    chatsRef.current = nextChats;
    activeChatIdRef.current = nextActiveChatId;

    const SpeechRecognition =
      typeof window !== "undefined"
        ? window.SpeechRecognition || window.webkitSpeechRecognition
        : undefined;

    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";
      recognition.onresult = (event) => {
        let finalTranscript = "";
        let interimTranscript = "";

        for (let i = event.resultIndex; i < event.results.length; i += 1) {
          const transcript = event.results[i][0]?.transcript ?? "";
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }

        const merged = `${finalTranscript}${interimTranscript}`.trim();
        setInput(merged ? merged : inputRef.current);
      };
      recognition.onerror = () => {
        setIsListening(false);
      };
      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
      setSpeechSupported(true);
    }
  }, []);

  useEffect(() => {
    if (!mounted) return;
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(serializeChats(chats)));
  }, [chats, mounted]);

  useEffect(() => {
    if (!mounted || !activeChatId) return;
    localStorage.setItem(ACTIVE_CHAT_STORAGE_KEY, activeChatId);
  }, [activeChatId, mounted]);

  useEffect(() => {
    function saveBeforeUnload() {
      localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(serializeChats(chatsRef.current)));
      if (activeChatIdRef.current) {
        localStorage.setItem(ACTIVE_CHAT_STORAGE_KEY, activeChatIdRef.current);
      }
    }

    window.addEventListener("beforeunload", saveBeforeUnload);
    return () => window.removeEventListener("beforeunload", saveBeforeUnload);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chats, isSending]);

  useEffect(() => {
    return () => {
      streamRef.current?.abort();
      recognitionRef.current?.stop();
    };
  }, []);

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      if (!menuRef.current) return;
      if (menuRef.current.contains(event.target as Node)) return;
      setOpenMenuChatId(null);
    }

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, []);

  const activeChat = chats.find((chat) => chat.id === activeChatId) ?? null;
  const visibleChats = [...chats].sort((left, right) => {
    if ((left.pinned ?? false) !== (right.pinned ?? false)) {
      return left.pinned ? -1 : 1;
    }
    if ((left.archived ?? false) !== (right.archived ?? false)) {
      return left.archived ? 1 : -1;
    }
    return 0;
  });

  const createNewChat = useCallback(() => {
    const nextChat = createChat();
    setChats((prev) => [nextChat, ...prev]);
    setActiveChatId(nextChat.id);
    chatsRef.current = [nextChat, ...chatsRef.current];
    activeChatIdRef.current = nextChat.id;
    setPendingAttachments([]);
    setInput("");
    setAttachmentError(null);
    setOpenMenuChatId(null);
  }, []);

  const deleteChat = useCallback((chatId: string) => {
    const remainingChats = chatsRef.current.filter((chat) => chat.id !== chatId);
    const nextChats = remainingChats.length > 0 ? remainingChats : [createChat()];
    const nextActiveChatId =
      activeChatIdRef.current === chatId
        ? nextChats[0].id
        : activeChatIdRef.current && nextChats.some((chat) => chat.id === activeChatIdRef.current)
          ? activeChatIdRef.current
          : nextChats[0].id;

    chatsRef.current = nextChats;
    activeChatIdRef.current = nextActiveChatId;
    setChats(nextChats);
    setActiveChatId(nextActiveChatId);
    setPendingAttachments([]);
    setAttachmentError(null);
    setOpenMenuChatId(null);
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(nextChats));
    localStorage.setItem(ACTIVE_CHAT_STORAGE_KEY, nextActiveChatId);
  }, []);

  const renameChat = useCallback((chatId: string) => {
    const chat = chatsRef.current.find((item) => item.id === chatId);
    if (!chat) return;

    const nextTitle = window.prompt("Rename chat", chat.title)?.trim();
    if (!nextTitle) return;

    setChats((prev) =>
      prev.map((item) => (item.id === chatId ? { ...item, title: nextTitle } : item))
    );
    chatsRef.current = chatsRef.current.map((item) =>
      item.id === chatId ? { ...item, title: nextTitle } : item
    );
    setOpenMenuChatId(null);
  }, []);

  const togglePinChat = useCallback((chatId: string) => {
    setChats((prev) =>
      prev.map((chat) => (chat.id === chatId ? { ...chat, pinned: !chat.pinned } : chat))
    );
    chatsRef.current = chatsRef.current.map((chat) =>
      chat.id === chatId ? { ...chat, pinned: !chat.pinned } : chat
    );
    setOpenMenuChatId(null);
  }, []);

  const toggleArchiveChat = useCallback((chatId: string) => {
    setChats((prev) =>
      prev.map((chat) => (chat.id === chatId ? { ...chat, archived: !chat.archived } : chat))
    );
    chatsRef.current = chatsRef.current.map((chat) =>
      chat.id === chatId ? { ...chat, archived: !chat.archived } : chat
    );
    setOpenMenuChatId(null);
  }, []);

  const shareChat = useCallback(async (chatId: string) => {
    const chat = chatsRef.current.find((item) => item.id === chatId);
    if (!chat) return;

    const shareText = `${chat.title}\n${window.location.href}`;
    try {
      await navigator.clipboard.writeText(shareText);
      setAttachmentError("Chat link copied to clipboard.");
    } catch {
      setAttachmentError("Share is not available in this browser.");
    }
    setOpenMenuChatId(null);
  }, []);

  const toggleListening = useCallback(() => {
    if (!recognitionRef.current) return;

    if (isListening) {
      recognitionRef.current.stop();
      return;
    }

    setAttachmentError(null);
    setIsListening(true);
    recognitionRef.current.start();
  }, [isListening]);

  const uploadAttachment = useCallback(async (file: File) => {
    const targetChat = ensureChat();
    const formData = new FormData();
    formData.append("session_id", targetChat.sessionId);
    formData.append("file", file);

    const response = await fetch(`${API_BASE}/chat/attachments`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
      throw new Error(payload?.detail || "Attachment upload failed.");
    }

    const data = (await response.json()) as UploadedAttachmentResponse;
    return {
      id: data.id,
      name: data.name,
      mimeType: data.mime_type,
      preview: data.preview,
    } satisfies Attachment;
  }, [ensureChat]);

  const removePendingAttachment = useCallback((attachmentId: string) => {
    setPendingAttachments((prev) => prev.filter((attachment) => attachment.id !== attachmentId));
  }, []);

  const sendMessage = useCallback(
    async (userMessage: string, attachments: Attachment[] = pendingAttachments) => {
      const userText = userMessage.trim();
      if (!userText && attachments.length === 0) {
        return;
      }

      const targetChat = ensureChat();
      const chatId = targetChat.id;
      const sessionId = targetChat.sessionId;
      const userMessageId = createId();
      const assistantMessageId = createId();
      const title = userText ? makeTitle(userText) : targetChat.title;

      setInput("");
      setPendingAttachments([]);
      setIsSending(true);
      setAttachmentError(null);

      updateChat(chatId, (chat) => ({
        ...chat,
        title:
          chat.messages.some((message) => message.role === "user") || !userText ? chat.title : title,
        messages: [
          ...chat.messages,
          {
            id: userMessageId,
            role: "user",
            content: userText || "Attached files",
            attachments,
          },
          {
            id: assistantMessageId,
            role: "assistant",
            content: "",
          },
        ],
      }));

      const url = new URL(`${API_BASE}/chat/stream`, window.location.origin);
      for (const attachment of attachments) {
        url.searchParams.append("attachment_ids", attachment.id);
      }

      streamRef.current?.abort();
      const abortController = new AbortController();
      streamRef.current = abortController;

      let accumulated = "";
      let buffer = "";

      try {
        const response = await fetch(url.toString(), {
          method: "POST",
          headers: {
            Accept: "text/event-stream",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: userText || "Summarize the attached files.",
            session_id: sessionId,
          }),
          signal: abortController.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP error: ${response.status}`);
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          setIsSending(false);
          return;
        }

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split("\n\n");
          buffer = events.pop() ?? "";

          for (const eventChunk of events) {
            const dataLines = eventChunk
              .split("\n")
              .filter((line) => line.startsWith("data:"))
              .map((line) => line.slice(5).trim())
              .join("");

            if (!dataLines) {
              continue;
            }

            const payload = JSON.parse(dataLines) as {
              delta?: string;
              done?: boolean;
            };

            if (payload.delta) {
              accumulated += payload.delta;
              updateChat(chatId, (chat) => ({
                ...chat,
                messages: chat.messages.map((message, index, messages) =>
                  index === messages.length - 1 && message.id === assistantMessageId
                    ? { ...message, content: accumulated }
                    : message
                ),
              }));
            }
          }
        }

        setIsSending(false);
        setTimeout(() => speak(accumulated), 200);
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        console.error("Chat error:", error);
        setIsSending(false);
        updateChat(chatId, (chat) => ({
          ...chat,
          messages: chat.messages.map((message, index, messages) =>
            index === messages.length - 1 && message.id === assistantMessageId
              ? {
                  ...message,
                  content: "I couldn't complete that request. Please try again.",
                }
              : message
          ),
        }));
      } finally {
        if (streamRef.current === abortController) {
          streamRef.current = null;
        }
      }
    },
    [ensureChat, pendingAttachments, speak, updateChat]
  );

  const handleImageSelect = useCallback(
    (file: File) => {
      const targetChat = ensureChat();
      const chatId = targetChat.id;
      const imageUrl = URL.createObjectURL(file);
      const title = makeTitle(file.name || "Image question");

      updateChat(chatId, (chat) => ({
        ...chat,
        title: chat.messages.some((message) => message.role === "user") ? chat.title : title,
        messages: [
          ...chat.messages,
          {
            id: createId(),
            role: "user",
            content: "",
            imageUrl,
            imageName: file.name,
            messageType: "pending_image",
            pendingFile: file,
            attachments: [],
          },
        ],
      }));
    },
    [ensureChat, updateChat]
  );

  const processImage = useCallback(
    async (messageId: string) => {
      const chatId = activeChatIdRef.current;
      if (!chatId) return;

      const chat = chatsRef.current.find((item) => item.id === chatId);
      const targetMessage = chat?.messages.find((message) => message.id === messageId);
      const file = targetMessage?.pendingFile;
      if (!file || !chat) return;

      const assistantMessageId = createId();

      updateChat(chatId, (currentChat) => ({
        ...currentChat,
        messages: [
          ...currentChat.messages.map((message): Message =>
            message.id === messageId
              ? ({
                  ...message,
                  messageType: "image",
                  pendingFile: null,
                } satisfies Message)
              : message
          ),
          ({
            id: assistantMessageId,
            role: "assistant",
            content: "Reading your image...",
            messageType: "text",
          } satisfies Message),
        ],
      }));

      setAttachmentError(null);
      setIsSending(true);

      try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("session_id", chat.sessionId);

        const response = await fetch(`${API_BASE}/chat/image`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`HTTP error: ${response.status}`);
        }

        const data = (await response.json()) as ImageChatResponse;
        updateChat(chatId, (currentChat) => ({
          ...currentChat,
          messages: currentChat.messages.map((message) =>
            message.id === assistantMessageId
              ? {
                  ...message,
                  content: data.answer,
                  extractedText: data.extracted_text?.trim() || undefined,
                }
              : message
          ),
        }));

        setTimeout(() => speak(data.answer), 200);
      } catch (error) {
        console.error("Image chat error:", error);
        updateChat(chatId, (currentChat) => ({
          ...currentChat,
          messages: currentChat.messages.map((message) =>
            message.id === assistantMessageId
              ? {
                  ...message,
                  content: "Could not process the image. Please type your question directly.",
                }
              : message
          ),
        }));
      } finally {
        setIsSending(false);
      }
    },
    [speak, updateChat]
  );

  const cancelImage = useCallback(
    (messageId: string) => {
      const chatId = activeChatIdRef.current;
      if (!chatId) return;

      updateChat(chatId, (chat) => {
        const targetMessage = chat.messages.find((message) => message.id === messageId);
        if (targetMessage?.imageUrl) {
          URL.revokeObjectURL(targetMessage.imageUrl);
        }

        return {
          ...chat,
          messages: chat.messages.filter((message) => message.id !== messageId),
        };
      });
    },
    [updateChat]
  );

  const handleAttachFiles = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.target.files ?? []);
      if (files.length === 0) return;

      setAttachmentError(null);

      try {
        const imageTypes = ["image/jpeg", "image/png", "image/webp", "image/gif"];
        const nonImageFiles = files.filter((file) => !imageTypes.includes(file.type));
        const imageFiles = files.filter((file) => imageTypes.includes(file.type));

        if (nonImageFiles.length > 0) {
          const uploaded = await Promise.all(nonImageFiles.map((file) => uploadAttachment(file)));
          setPendingAttachments((prev) => [...prev, ...uploaded]);
        }

        for (const imageFile of imageFiles) {
          handleImageSelect(imageFile);
        }
      } catch (error) {
        setAttachmentError(error instanceof Error ? error.message : "Attachment upload failed.");
      } finally {
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }
    },
    [handleImageSelect, uploadAttachment]
  );

  const formatMessage = (content: string) => {
    if (!content) return null;

    return content.split("\n").map((line, index, lines) => (
      <span key={`${index}-${line.slice(0, 20)}`}>
        {line || "\u00A0"}
        {index < lines.length - 1 && <br />}
      </span>
    ));
  };

  if (!mounted || !activeChat) {
    return null;
  }

  return (
    <div className="flex h-screen bg-[#0b0d12] text-zinc-100">
      <aside className="hidden w-72 shrink-0 border-r border-white/10 bg-[#0f1117] px-4 py-5 md:flex md:flex-col">
        <button
          onClick={createNewChat}
          className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-left text-sm font-medium text-white transition hover:bg-white/10"
        >
          New chat
        </button>

        <div className="mt-6 text-xs font-medium uppercase tracking-[0.18em] text-zinc-500">
          Chats
        </div>

        <div className="mt-3 flex-1 space-y-2 overflow-y-auto">
          {visibleChats.map((chat) => (
            <div
              key={chat.id}
              className={`group relative flex items-center gap-2 rounded-2xl px-2 py-2 transition ${
                chat.id === activeChatId ? "bg-white/10" : "hover:bg-white/5"
              }`}
            >
              <button
                onClick={() => {
                  setActiveChatId(chat.id);
                  setPendingAttachments([]);
                  setAttachmentError(null);
                  setOpenMenuChatId(null);
                }}
                className={`min-w-0 flex-1 rounded-xl px-2 py-2 text-left text-sm transition ${
                  chat.id === activeChatId ? "text-white" : "text-zinc-400 group-hover:text-white"
                }`}
              >
                <div className="flex items-center gap-2">
                  <div className="truncate font-medium">{chat.title}</div>
                  {chat.pinned && (
                    <span className="rounded-full bg-white/10 px-2 py-0.5 text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                      Pinned
                    </span>
                  )}
                  {chat.archived && (
                    <span className="rounded-full bg-white/10 px-2 py-0.5 text-[10px] uppercase tracking-[0.14em] text-zinc-500">
                      Archived
                    </span>
                  )}
                </div>
              </button>
              <button
                onClick={(event) => {
                  event.stopPropagation();
                  setOpenMenuChatId((current) => (current === chat.id ? null : chat.id));
                }}
                className={`shrink-0 rounded-xl px-2 py-1 text-sm text-zinc-500 transition hover:bg-white/10 hover:text-white ${
                  openMenuChatId === chat.id ? "bg-white/10 text-white" : "opacity-0 group-hover:opacity-100"
                }`}
                aria-label={`Open actions for ${chat.title}`}
                title="Chat actions"
              >
                ...
              </button>

              {openMenuChatId === chat.id && (
                <div
                  ref={menuRef}
                  className="absolute right-2 top-12 z-20 w-40 rounded-2xl border border-white/10 bg-[#171a22] p-1 shadow-[0_24px_60px_-28px_rgba(0,0,0,0.85)]"
                >
                  <button
                    onClick={() => renameChat(chat.id)}
                    className="w-full rounded-xl px-3 py-2 text-left text-sm text-zinc-200 transition hover:bg-white/10"
                  >
                    Rename
                  </button>
                  <button
                    onClick={() => deleteChat(chat.id)}
                    className="w-full rounded-xl px-3 py-2 text-left text-sm text-rose-300 transition hover:bg-white/10"
                  >
                    Delete
                  </button>
                  <button
                    onClick={() => togglePinChat(chat.id)}
                    className="w-full rounded-xl px-3 py-2 text-left text-sm text-zinc-200 transition hover:bg-white/10"
                  >
                    Pin
                  </button>
                  <button
                    onClick={() => toggleArchiveChat(chat.id)}
                    className="w-full rounded-xl px-3 py-2 text-left text-sm text-zinc-200 transition hover:bg-white/10"
                  >
                    Archive
                  </button>
                  <button
                    onClick={() => void shareChat(chat.id)}
                    className="w-full rounded-xl px-3 py-2 text-left text-sm text-zinc-200 transition hover:bg-white/10"
                  >
                    Share
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>

        <button
          onClick={() => setVoiceOn((value) => !value)}
          className={`rounded-2xl border px-4 py-3 text-sm font-medium transition ${
            voiceOn
              ? "border-emerald-400/40 bg-emerald-400/10 text-emerald-200"
              : "border-white/10 bg-white/5 text-zinc-300 hover:bg-white/10"
          }`}
        >
          Speaker voice {voiceOn ? "on" : "off"}
        </button>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="border-b border-white/10 bg-[#0b0d12]/90 px-4 py-4 backdrop-blur md:px-8">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h1 className="text-base font-semibold text-white md:text-lg">{activeChat.title}</h1>
              <p className="text-sm text-zinc-500">Bank assistant with streaming answers and per-message files.</p>
            </div>
            <button
              onClick={createNewChat}
              className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-zinc-200 transition hover:bg-white/10 md:hidden"
            >
              New chat
            </button>
          </div>
        </header>

        <section className="flex-1 overflow-y-auto px-4 py-6 md:px-8">
          <div className="mx-auto flex w-full max-w-4xl flex-col gap-4">
            {activeChat.messages.map((message) => (
              <div
                key={message.id}
                className={`max-w-[88%] rounded-[24px] px-4 py-3 text-sm shadow-[0_18px_50px_-30px_rgba(0,0,0,0.7)] md:px-5 md:py-4 ${
                  message.role === "user"
                    ? "ml-auto bg-[#e8eaee] text-[#14161b]"
                    : "border border-white/10 bg-[#12151c] text-zinc-100"
                }`}
              >
                {message.messageType !== "pending_image" && (
                  <div
                    className="leading-6"
                    style={{
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                    }}
                  >
                    {message.content ? (
                      message.role === "assistant" ? (
                        formatMessage(message.content)
                      ) : (
                        message.content
                      )
                    ) : (
                      <span className="text-zinc-500">Thinking...</span>
                    )}
                  </div>
                )}

                {message.imageUrl && (
                  <div className="mt-3">
                    <Image
                      src={message.imageUrl}
                      alt={message.imageName || "uploaded image"}
                      width={200}
                      height={200}
                      className="block max-h-[200px] max-w-[200px] rounded-xl object-cover"
                      unoptimized
                    />
                  </div>
                )}

                {message.messageType === "pending_image" && (
                  <div className="mt-3">
                    {message.imageUrl && (
                      <div className="mb-3">
                        <Image
                          src={message.imageUrl}
                          alt={message.imageName || "uploaded image"}
                          width={200}
                          height={200}
                          className="block max-h-[200px] max-w-[200px] rounded-xl object-cover"
                          unoptimized
                        />
                      </div>
                    )}
                    <div className="text-[13px] text-zinc-400">
                      What would you like me to do with this image?
                    </div>
                    <div className="mt-3 flex gap-2">
                      <button
                        onClick={() => void processImage(message.id)}
                        className="rounded-lg bg-sky-500 px-3 py-2 text-[13px] text-white transition hover:bg-sky-400"
                      >
                        Read and Answer Questions
                      </button>
                      <button
                        onClick={() => cancelImage(message.id)}
                        className="rounded-lg border border-white/10 px-3 py-2 text-[13px] text-zinc-400 transition hover:bg-white/5 hover:text-white"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}

                {message.extractedText && (
                  <div className="mt-3 text-[11px] italic text-zinc-500">
                    Extracted: {message.extractedText}
                  </div>
                )}

                {message.attachments && message.attachments.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {message.attachments.map((attachment) => (
                      <div
                        key={attachment.id}
                        className="rounded-2xl border border-black/10 bg-black/5 px-3 py-2 text-xs text-current/80"
                        title={attachment.preview}
                      >
                        <div className="font-medium">{attachment.name}</div>
                        {attachment.preview && (
                          <div className="mt-1 max-w-56 truncate text-current/60">{attachment.preview}</div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        </section>

        <footer className="border-t border-white/10 bg-[#0b0d12] px-4 py-4 md:px-8 md:py-6">
          <div className="mx-auto w-full max-w-4xl">
            {pendingAttachments.length > 0 && (
              <div className="mb-3 flex flex-wrap gap-2">
                {pendingAttachments.map((attachment) => (
                  <button
                    key={attachment.id}
                    onClick={() => removePendingAttachment(attachment.id)}
                    className="rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-left text-xs text-zinc-200 transition hover:bg-white/10"
                  >
                    <div className="flex items-center gap-2">
                      <span className="max-w-48 truncate font-medium">{attachment.name}</span>
                      <span className="text-zinc-500">x</span>
                    </div>
                    {attachment.preview && (
                      <div className="mt-1 max-w-56 truncate text-zinc-400">{attachment.preview}</div>
                    )}
                  </button>
                ))}
              </div>
            )}

            {attachmentError && <div className="mb-3 text-sm text-rose-300">{attachmentError}</div>}

            <div className="rounded-[28px] border border-white/10 bg-[#11141b] p-2 shadow-[0_20px_60px_-30px_rgba(0,0,0,0.85)]">
              <div className="flex items-end gap-2">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-white/5 text-lg text-zinc-300 transition hover:bg-white/10 hover:text-white"
                  aria-label="Attach files"
                >
                  +
                </button>

                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  multiple
                  accept=".pdf,.txt,.md,.csv,.json,.png,.jpg,.jpeg,.webp,.gif"
                  onChange={handleAttachFiles}
                />

                <div className="flex min-h-11 flex-1 items-center">
                  <input
                    className="h-11 w-full bg-transparent px-3 text-sm text-zinc-100 outline-none placeholder:text-zinc-500"
                    placeholder="Message the bank assistant"
                    value={input}
                    onChange={(event) => setInput(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        void sendMessage(input, pendingAttachments);
                      }
                    }}
                  />
                </div>

                <button
                  onClick={toggleListening}
                  disabled={!speechSupported}
                  className={`flex h-11 shrink-0 items-center justify-center rounded-2xl px-4 text-sm font-medium transition ${
                    isListening
                      ? "bg-rose-500 text-white"
                      : "bg-white/5 text-zinc-300 hover:bg-white/10 hover:text-white disabled:cursor-not-allowed disabled:bg-white/[0.03] disabled:text-zinc-600"
                  }`}
                  aria-label={isListening ? "Stop listening" : "Start listening"}
                >
                  {isListening ? "Listening..." : "Mic"}
                </button>

                <button
                  onClick={() => void sendMessage(input, pendingAttachments)}
                  disabled={isSending}
                  className="flex h-11 min-w-11 shrink-0 items-center justify-center rounded-2xl bg-white text-sm font-medium text-[#14161b] transition hover:bg-zinc-200 disabled:cursor-not-allowed disabled:bg-zinc-700 disabled:text-zinc-400"
                  aria-label="Send message"
                >
                  Send
                </button>
              </div>
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
}