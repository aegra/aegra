import { Annotation, MessagesAnnotation } from "@langchain/langgraph";

/**
 * Chat state extending the built-in MessagesAnnotation.
 *
 * MessagesAnnotation provides a `messages` channel with automatic
 * message accumulation (new messages are appended, not replaced).
 */
export const ChatState = Annotation.Root({
  ...MessagesAnnotation.spec,
});

export type ChatStateType = typeof ChatState.State;
