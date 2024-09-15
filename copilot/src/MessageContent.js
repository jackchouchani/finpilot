import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Typography, Paper } from '@mui/material';

const MessageContent = ({ content }) => {
    // Remplacer les \n par des retours à la ligne HTML
    const formattedContent = content.replace(/\\n/g, '\n');

    return (
        <Paper style={{ padding: '10px', marginTop: '10px', maxWidth: '100%', overflowX: 'auto' }}>
            <ReactMarkdown
                components={{
                    h1: ({ node, ...props }) => <Typography variant="h4" gutterBottom {...props} />,
                    h2: ({ node, ...props }) => <Typography variant="h5" gutterBottom {...props} />,
                    h3: ({ node, ...props }) => <Typography variant="h6" gutterBottom {...props} />,
                    p: ({ node, ...props }) => <Typography paragraph {...props} />,
                    li: ({ node, ...props }) => <Typography component="li" style={{ marginLeft: 20 }} {...props} />,
                    ul: ({ node, ...props }) => <ul style={{ paddingLeft: 20 }} {...props} />,
                    code: ({ node, inline, ...props }) =>
                        inline ? (
                            <code style={{ backgroundColor: '#f0f0f0', padding: '2px 4px', borderRadius: '4px' }} {...props} />
                        ) : (
                            <Paper style={{ padding: '10px', backgroundColor: '#f5f5f5', overflowX: 'auto' }}>
                                <code {...props} />
                            </Paper>
                        ),
                }}
            >
                {formattedContent}
            </ReactMarkdown>
        </Paper>
    );
};

export default MessageContent;