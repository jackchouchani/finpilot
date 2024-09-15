import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Typography, Paper, Box } from '@mui/material';

const MessageContent = ({ content }) => {
    const isJSON = (str) => {
        try {
            const parsed = JSON.parse(str);
            // Vérifie si c'est un objet JSON "normal" et non une chaîne transformée en objet
            return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed) && Object.keys(parsed).length > 0 && !Object.keys(parsed).every(key => !isNaN(parseInt(key)));
        } catch (e) {
            return false;
        }
    };

    const isStringObject = (str) => {
        try {
            const parsed = JSON.parse(str);
            return typeof parsed === 'object' && parsed !== null && Object.keys(parsed).every(key => !isNaN(parseInt(key)));
        } catch (e) {
            return false;
        }
    };

    if (isJSON(content)) {
        const jsonData = JSON.parse(content);
        return (
            <Paper sx={{ p: 2, mt: 1, maxWidth: '100%', overflowX: 'auto' }}>
                <pre>{JSON.stringify(jsonData, null, 2)}</pre>
            </Paper>
        );
    } else if (isStringObject(content)) {
        const stringContent = Object.values(JSON.parse(content)).join('');
        return (
            <Paper sx={{ p: 2, mt: 1, maxWidth: '100%', overflowX: 'auto' }}>
                <Typography>{stringContent}</Typography>
            </Paper>
        );
    }

    // Pour le contenu non-JSON, on utilise ReactMarkdown comme avant
    return (
        <Paper sx={{ p: 2, mt: 1, maxWidth: '100%', overflowX: 'auto' }}>
            <ReactMarkdown
                components={{
                    h1: ({ node, ...props }) => <Typography variant="h4" gutterBottom {...props} />,
                    h2: ({ node, ...props }) => <Typography variant="h5" gutterBottom {...props} />,
                    h3: ({ node, ...props }) => <Typography variant="h6" gutterBottom {...props} />,
                    p: ({ node, ...props }) => <Typography paragraph {...props} />,
                    li: ({ node, ...props }) => <Typography component="li" sx={{ ml: 2 }} {...props} />,
                    ul: ({ node, ...props }) => <Box component="ul" sx={{ pl: 2 }} {...props} />,
                    code: ({ node, inline, ...props }) =>
                        inline ? (
                            <Box component="code" sx={{ bgcolor: 'grey.100', p: 0.5, borderRadius: 1 }} {...props} />
                        ) : (
                            <Box component="pre" sx={{ p: 1, bgcolor: 'grey.100', borderRadius: 1, overflow: 'auto' }}>
                                <code {...props} />
                            </Box>
                        ),
                }}
            >
                {content}
            </ReactMarkdown>
        </Paper>
    );
};

export default MessageContent;