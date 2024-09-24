import React, { useRef, useEffect, useCallback } from 'react';
import { Box, TextField, Button, List, ListItem, ListItemText, Paper } from '@mui/material';
import MessageContent from './MessageContent';

function ChatBox({ messages, input, setInput, handleSubmit, loading }) {
    const messagesEndRef = useRef(null);

    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, []);

    useEffect(scrollToBottom, [messages]);

    const handleInputChange = useCallback((e) => {
        setInput(e.target.value);
    }, [setInput]);

    const inputRef = useRef(null);

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <Paper sx={{ flexGrow: 1, overflow: 'auto', mb: 2, p: 2 }}>
                <List>
                    {messages.map((message, index) => (
                        <ListItem key={index} alignItems="flex-start">
                            <ListItemText
                                primary={message.role === 'user' ? 'Vous' : 'IA'}
                                secondary={
                                    <>
                                        <MessageContent content={message.content} />
                                        {message.graphs && message.graphs.map((graph, graphIndex) => (
                                            <img
                                                key={graphIndex}
                                                src={`data:image/png;base64,${graph}`}
                                                alt={`Portfolio Graph ${graphIndex + 1}`}
                                                style={{ maxWidth: '100%', marginTop: '10px' }}
                                            />
                                        ))}
                                        {message.graph && (
                                            <img
                                                src={`data:image/png;base64,${message.graph}`}
                                                alt="Graph"
                                                style={{ maxWidth: '100%', marginTop: '10px' }}
                                            />
                                        )}
                                    </>
                                }
                            />
                        </ListItem>
                    ))}
                    <div ref={messagesEndRef} />
                </List>
            </Paper>
            <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', position: 'sticky', bottom: 0, bgcolor: 'background.paper' }}>
                <TextField
                    fullWidth
                    placeholder="Tapez votre message ici..."
                    variant="outlined"
                    disabled={loading}
                    inputRef={inputRef}
                />
                <Button type="submit" variant="contained" disabled={loading || !input.trim()}>
                    Envoyer
                </Button>
            </Box>
        </Box>
    );
}

export default ChatBox;