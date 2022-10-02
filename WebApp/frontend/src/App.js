import React from 'react';
import {
  Center,
  Text,
  Heading,
  VStack,
  Button,
  Input,
  HStack,
  Container,
  SimpleGrid,
  Box,
  Image,
  Spinner,
  ChakraProvider,
} from '@chakra-ui/react';

import { useState, useEffect } from 'react';

function App() {
  const [isSelected, setIsSelected] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [allPhotos, setAllPhotos] = useState([]);
  const [uploadSuccessful, setUploadSuccessful] = useState(false);
  const [showSpinner, setShowSpinner] = useState(false);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/images')
      .then(res => res.json())
      .then(data => {
        setAllPhotos(data);
      });
  }, [uploadSuccessful]);

  const onInputChange = e => {
    setSelectedFile(e.target.files[0]);
    setIsSelected(true);
  };

  const onFileUpload = e => {
    setShowSpinner(true);
    const formData = new FormData();
    formData.append('file', selectedFile, selectedFile.name);

    fetch('http://127.0.0.1:8000/images', {
      method: 'POST',
      body: formData,
    })
      .then(res => res.json())
      .then(data => {
        console.log('Upload Successful!!!');
        setUploadSuccessful(!uploadSuccessful);
        setShowSpinner(false);
      });
  };

  return (
    <ChakraProvider>
      <Center bg="black" color="white" padding={0}>
        <VStack spacing={0} padding={0}>
          <Heading size="2xl" padding={0} spacing={7}>
            Mitotic Image Classifier
          </Heading>
          <Text> Upload your Image!</Text>
          <HStack>
            <input type="file" onChange={onInputChange} onClick={null} />
            <Button
              size={'lg'}
              colorScheme="red"
              onClick={onFileUpload}
              isDisabled={!isSelected}
            >
              Upload Image
            </Button>
            {showSpinner ? (
              <Center>
                {' '}
                <Spinner size={'xl'} />
              </Center>
            ) : null}
          </HStack>
          <Heading>Your Photos</Heading>
          <SimpleGrid columns={3} spacing={10}>
            {allPhotos.length !== 0 &&
              allPhotos.map(photo => {
                return <Image src={photo['image_url']}></Image>;
              })}
          </SimpleGrid>
        </VStack>
      </Center>
    </ChakraProvider>
  );
}

export default App;
