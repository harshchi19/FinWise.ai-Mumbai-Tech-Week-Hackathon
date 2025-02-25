import OpenAI from "openai";
import { NextResponse } from "next/server";

const openai = new OpenAI({
    apiKey: process.env.NEXT_PUBLIC_OPENAI_API_KEY,
    baseURL: "https://api.aimlapi.com",
});

const avatarPersonas = {
    1: {
        systemPrompt: "You are Tony Stark (Ironman). Respond with a mix of technical genius, witty sarcasm, and confidence. Use tech-related analogies and speak as if you're the smartest person in the room.",
    },
    2: {
        systemPrompt: "You are Thor, the God of Thunder. Speak with a regal, old-world tone, using dramatic language and occasional references to Asgardian culture. Show strength and nobility.",
    },
    3: {
        systemPrompt: "You are Steve Rogers (Captain America). Respond with moral clarity, leadership, and a straightforward, principled approach. Use inspirational and direct language.",
    },
    4: {
        systemPrompt: "You are Natasha Romanoff (Black Widow). Respond with intelligence, strategic thinking, and a slight edge of mystery. Use concise, precise language with occasional dark humor.",
    }
};

export async function POST(req) {
    try {
        const { message, avatar } = await req.json();

        if (!message || typeof message !== 'string') {
            return NextResponse.json(
                { error: "Invalid message format" },
                { status: 400 }
            );
        }

        if (!avatarPersonas[avatar]) {
            return NextResponse.json(
                { error: "Invalid avatar selected" },
                { status: 400 }
            );
        }

        const chatCompletion = await openai.chat.completions.create({
            model: "gpt-3.5-turbo",
            messages: [
                {
                    role: "system",
                    content: avatarPersonas[avatar].systemPrompt
                },
                {
                    role: "user",
                    content: message
                }
            ],
            temperature: 0.7,
            max_tokens: 150,
        });

        const rawReply = chatCompletion.choices[0].message.content?.trim() ||
            "I'm unable to respond at the moment.";

        const reply = `${rawReply}`;

        return NextResponse.json({
            reply,
            avatar: avatar
        });
    } catch (error) {
        console.error("Chat API Error:", error);
        return NextResponse.json(
            { error: "Failed to process your message" },
            { status: 500 }
        );
    }
}