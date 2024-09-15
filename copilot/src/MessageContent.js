import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Typography, Paper, Box } from '@mui/material';

const MessageContent = ({ content }) => {
    // Fonction pour vérifier si le contenu est valide
    const isValidContent = (content) => {
        return content !== null && content !== undefined && typeof content === 'string';
    };

    const isMarkdown = (str) => {
        return str.startsWith('# ') || str.includes('\n# ') || str.includes('\n## ');
    };

    // Fonction pour formater le contenu
    const formatContent = (text) => {
        if (!isValidContent(text)) return '';
        return text.replace(/\\n/g, '\n').replace(/\n(?!\n)/g, '\n\n');
    };

    // Fonction pour vérifier si c'est du JSON valide
    const isJSON = (str) => {
        if (!isValidContent(str)) return false;
        try {
            const parsed = JSON.parse(str);
            return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed) && Object.keys(parsed).length > 0;
        } catch (e) {
            return false;
        }
    };

    // Fonction pour vérifier si c'est un objet string
    const isStringObject = (str) => {
        if (!isValidContent(str)) return false;
        try {
            const parsed = JSON.parse(str);
            return typeof parsed === 'object' && parsed !== null && Object.keys(parsed).every(key => !isNaN(parseInt(key)));
        } catch (e) {
            return false;
        }
    };

    // Gestion du contenu invalide
    if (!isValidContent(content)) {
        return (
            <Paper sx={{ p: 2, mt: 1, maxWidth: '100%', overflowX: 'auto' }}>
                <Typography color="error">Contenu invalide ou non disponible</Typography>
            </Paper>
        );
    }

    // Traitement du contenu JSON
    if (isJSON(content)) {
        try {
            const jsonData = JSON.parse(content);
            return (
                <Paper sx={{ p: 2, mt: 1, maxWidth: '100%', overflowX: 'auto' }}>
                    <pre>{JSON.stringify(jsonData, null, 2)}</pre>
                </Paper>
            );
        } catch (error) {
            console.error("Erreur lors du parsing JSON:", error);
            return (
                <Paper sx={{ p: 2, mt: 1, maxWidth: '100%', overflowX: 'auto' }}>
                    <Typography color="error">Erreur lors de l'affichage du contenu JSON</Typography>
                </Paper>
            );
        }
    }

    // Traitement du contenu string object
    if (isStringObject(content)) {
        try {
            const stringContent = Object.values(JSON.parse(content)).join('');
            return (
                <Paper sx={{ p: 2, mt: 1, maxWidth: '100%', overflowX: 'auto' }}>
                    <Typography>{formatContent(stringContent)}</Typography>
                </Paper>
            );
        } catch (error) {
            console.error("Erreur lors du parsing de l'objet string:", error);
            return (
                <Paper sx={{ p: 2, mt: 1, maxWidth: '100%', overflowX: 'auto' }}>
                    <Typography color="error">Erreur lors de l'affichage du contenu</Typography>
                </Paper>
            );
        }
    }
    // Traitement du contenu Markdown
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
                    img: ({ node, ...props }) => <Box component="img" sx={{ maxWidth: '100%', height: 'auto' }} {...props} />,
                }}
            >
                {formatContent(content)}
            </ReactMarkdown>
        </Paper>
    );
};

export default MessageContent;