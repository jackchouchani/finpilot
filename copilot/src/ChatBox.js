import React, { useRef, useEffect } from 'react';
import { Box, TextField, Button, List, ListItem, ListItemText, Paper } from '@mui/material';
import MessageContent from './MessageContent';

function ChatBox({ messages, input, setInput, handleSubmit, loading }) {
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages]);

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <Paper sx={{ flexGrow: 1, overflow: 'auto', mb: 2, p: 2 }}>
                <List>
                    {messages.slice().reverse().map((message, index) => (
                        <ListItem key={index} alignItems="flex-start">
                            <ListItemText
                                primary={message.role === 'user' ? 'Vous' : 'IA'}
                                secondary={<MessageContent content={message.content} />}
                            />
                        </ListItem>
                    ))}
                    <div ref={messagesEndRef} />
                </List>
            </Paper>
            <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', position: 'sticky', bottom: 0, bgcolor: 'background.paper' }}>
                <TextField
                    fullWidth
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Tapez votre message ici..."
                    variant="outlined"
                    disabled={loading}
                />
                <Button type="submit" variant="contained" disabled={loading || !input.trim()}>
                    Envoyer
                </Button>
            </Box>
        </Box>
    );
}

export default ChatBox;