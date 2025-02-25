import { Star, LayoutDashboard, Workflow, BotMessageSquare, Newspaper, CircleUserRound, Bot } from 'lucide-react'

export const sidebarData = [
    {
        category: "dashboard",
        items: [
            { title: "overview", route: "/overview", icon: LayoutDashboard },
            { title: "favourite", route: "/favourite", icon: Star },
        ],
    },
    {
        category: "user",
        items: [
            { title: "chatbot", route: "/user/chatbot", icon: BotMessageSquare },
            { title: "flow", route: "/user/flow", icon: Workflow },
            { title: "news", route: "/user/news", icon: Newspaper },
            { title: "profile", route: "/user/profile", icon: CircleUserRound },
        ],
    },
];
